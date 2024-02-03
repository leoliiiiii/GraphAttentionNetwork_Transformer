import json
import math
import numpy as np
import os
import sys
import tiktoken
import time
import torch
import torch.nn.functional as F
import wandb
from collections import deque
from collections.abc import Iterator
from contextlib import nullcontext
from einops import rearrange
from model import DecoderLM
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from typing import Callable
from utils import count_params, determine_device, enable_tf32



def random_batch_sampler(
    tokens: torch.LongTensor, device: str, batch_size: int, seq_len: int
) -> Iterator[torch.LongTensor]:
    """An infinite generator that samples batches of sequences from the tokens.

    Args:
        tokens: a 1d torch tensor of token ids
        device: the device to put the batch on
        batch_size: the batch size of the output tensor (B)
        seq_len: the sequence length of the output tensor (S)

    Returns:
        An infinite generator that samples batches of sequences from the
        tokens. Each batch has shape (B x S). Every sequence in the batch is
        a contiguous subsequence of x, sampled uniformly at random. The
        output tensor should be on the right device.
    """
    assert seq_len < len(tokens)
    while True:
        # Sample for starting positions
        start = torch.randint(0, len(tokens) - seq_len + 1, (batch_size,))
        seq = [tokens[s : s+seq_len] for s in start]
        # Stack all samples into a batch
        batch = torch.stack(seq).to(device)
        yield batch


def sequential_batch_sampler(
    tokens: torch.LongTensor, device: str, batch_size: int, seq_len: int
) -> Iterator[torch.LongTensor]:
    """A generator that yields batches of tokens.

    Args:
        tokens: a 1d torch tensor of token ids
        device: the device to put the batch on
        batch_size: the batch size of the output tensor (B)
        seq_len: the sequence length of the output tensor (S)

    Returns:
        A generator that yields a batch of tokens at a time. Each batch has
        shape (B x S). Every sequence in the batch is a contiguous subsequence
        of x in sequential order. The output tensor should be on the right
        device.
        
    Note: If the last batch is incomplete, which could happen when the number
        of tokens is not divisible by (batch_size * seq_len), you could drop
        the last batch.
    """
    batch_len = batch_size * seq_len
    batch_num = len(tokens) // batch_len
    batches = [tokens[i * batch_len : (i+1) * batch_len].reshape(batch_size, seq_len).to(device)
               for i in range(batch_num)]
    for batch in batches:
        yield batch


def cosine_lr_schedule(
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float,
    max_lr: float,
) -> Callable[[int], float]:
    def get_lr(t: int) -> float:
        """Outputs the learning rate at step t under the cosine schedule.

        Args:
            t: the current step number

        Returns:
            lr: learning rate at step t

        Hint: Question 2.1
        """

        assert max_lr >= min_lr >= 0.0
        assert num_training_steps >= num_warmup_steps >= 0

        if t < num_warmup_steps:
            lr = (t / num_warmup_steps) * max_lr
        elif t < num_training_steps:
            lr = min_lr + 0.5 * (max_lr - min_lr) * \
                 (1 + math.cos(math.pi * (t - num_warmup_steps) / (num_training_steps - num_warmup_steps)))
        else:  # t >= num_training_steps
            lr = min_lr
        return lr

    return get_lr


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def compute_language_modeling_loss(
    input_ids: torch.LongTensor, logits: torch.FloatTensor
) -> torch.FloatTensor:
    """Outputs the language modeling loss given input_ids and logits

    Args:
        input_ids: the input token ids
        logits: the next token logits produced by the language model

    Returns:
        loss: the mean cross entropy loss for next token prediction

    Hint: Think about what are the groundtruth labels for next token prediction.
    """

    # The true label is the actual next token
    labels = input_ids[:, 1:].clone()
    # There is no label for the last word and therefore no valid loss
    logits = logits[:, :-1, :].clone()
    # Reshaping labels and logits
    labels = labels.reshape(-1)
    logits = logits.reshape(labels.size(0), -1)
    ce = torch.nn.CrossEntropyLoss()
    loss = ce(logits, labels)
    return loss


def train(
    model: DecoderLM,
    batch_sampler: Iterator[torch.LongTensor],
    optimizer: torch.optim.Optimizer,
    lr_schedule: Callable[[int], float],
    autocast: torch.autocast | nullcontext = nullcontext(),
    num_training_steps: int = 0,
    grad_accumulation_steps: int = 1,
) -> None:
    """A training loop for the language model

    Args:
        model: the decoder LM
        batch_sampler: a generator that produces batches of token ids
        optimizer: an optimizer for gradient update
        lr_schedule: a callable that produces the learning at a step number
        autocast: a context manager that handles tensor casting (you do not need
          to care about this for your implementation)
        num_training_steps: number of steps to train for
        grad_accumulation_steps: number of "micro" training steps before each
          gradient update
    """
    # stores training losses for the 20 latest steps
    losses = deque(maxlen=20 * grad_accumulation_steps)

    for step in (pbar := trange(num_training_steps)):
        t0 = time.time()
        lr = lr_schedule(step)
        set_lr(optimizer, lr)

        for _ in range(grad_accumulation_steps):
            # TODO: sample a batch, generate logits and compute loss
            input_ids = next(batch_sampler)
            with autocast:
                logits = model(input_ids)
            loss = compute_language_modeling_loss(input_ids, logits)
            (loss / grad_accumulation_steps).backward()
            loss_f = loss.item()
            losses.append(loss_f)

        # TODO: update the model using the accumulated gradients
        optimizer.step()
        optimizer.zero_grad()
        loss_mean = np.mean(losses).item()

        FLOPs_per_step = (
            model.flops_per_token
            * input_ids.shape[0]
            * input_ids.shape[1]
            * grad_accumulation_steps
        )
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        pbar.set_postfix(
            {
                "train loss": f"{loss_mean:.2f}",
                "TFLOPS": f"{FLOPs_per_step / dt / 1e12:.1f}",
            }
        )
        wandb.log({"train-loss": loss_mean, "learning-rate": lr}, step=step)


@torch.inference_mode()
def evaluate(
    model: DecoderLM,
    batch_sampler: Iterator[torch.LongTensor],
    autocast: torch.autocast | nullcontext = nullcontext(),
) -> dict[str, float]:
    losses = []

    for input_ids in (pbar := tqdm(batch_sampler, desc="evaluating..")):
        with autocast:
            logits = model(input_ids)
        loss = compute_language_modeling_loss(input_ids, logits)
        losses.append(loss.item())

    # mean of the losses is the average negative log likelihood
    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)

    eval_results = {
        "val-loss": mean_loss,
        "val-perplexity": perplexity,
    }
    wandb.log(eval_results)
    return eval_results


def main():
    enable_tf32()

    # create an output directory and dump the configuration file
    assert len(sys.argv) > 1, "provide a configuration file"
    config = OmegaConf.load(sys.argv[1])
    os.makedirs(config.output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))
    print("#" * 40, OmegaConf.to_yaml(config).strip(), "#" * 40, sep="\n")
    wandb.init(project="llms-hw2", config=OmegaConf.to_container(config))

    # initialize tokenizer and model
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    print(f"model parameters = {count_params(model) / 1e6:.0f}M")

    # prepare data and data generator

    assert config.seq_len <= config.model_config.n_positions
    tokens = np.load("data/tokens.npz")
    train_tokens = torch.from_numpy(tokens["train"].astype(int))
    val_tokens = torch.from_numpy(tokens["val"].astype(int))

    train_sampler = random_batch_sampler(
        train_tokens, device, config.batch_size, config.seq_len
    )
    val_sampler = sequential_batch_sampler(
        val_tokens, device, config.batch_size, config.seq_len
    )
    print(f"train dataset tokens = {len(train_tokens) / 1e6:.0f}M")
    FLOPs = (
        model.flops_per_token
        * config.num_training_steps
        * config.grad_accumulation_steps
        * config.batch_size
        * config.seq_len
    )
    print(f"train FLOPs = {FLOPs:.2e}")

    # prepare optimizer and lr schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,  # will set this dynamically in the training loop
        betas=(0.9, 0.95),
        fused=device == "cuda",
    )
    lr_schedule = cosine_lr_schedule(
        config.num_warmup_steps, config.num_training_steps, config.min_lr, config.max_lr
    )
    autocast = (
        torch.autocast(
            device,
            dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32),
        )
        if device == "cuda"
        else nullcontext()
    )
    # training
    model.train()
    train(
        model,
        train_sampler,
        optimizer,
        lr_schedule,
        autocast,
        config.num_training_steps,
        config.grad_accumulation_steps,
    )

    # save the trained model
    model_path = os.path.join(config.output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"model saved to {model_path}")

    # evaluation
    model.eval()
    eval_results = evaluate(model, val_sampler, autocast)
    print("evaluation results:", json.dumps(eval_results))
    with open(os.path.join(config.output_dir, "eval.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
    print("done!")


if __name__ == "__main__":
    main()
