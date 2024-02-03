import argparse
import json
import os
import tiktoken
import torch
from model import DecoderLM
from omegaconf import OmegaConf
from tqdm import trange
from utils import determine_device, enable_tf32


def softmax_with_temperature(
    logits: torch.FloatTensor, temperature: float
) -> torch.FloatTensor:
    """Turns logits into probabilities under softmax (with temperature)

    Args:
        logits: a 2d torch tensor of token ids (B x V)
        temperature: temperature of the softmax function

    Returns:
        a 2d torch tensor of token probabilities (B x V)
    """

    # to avoid division by 0
    temperature = max(temperature, 1e-5)

    return torch.nn.functional.softmax(logits / temperature, dim=-1)


@torch.inference_mode()
def generate(
    model: DecoderLM,
    device: str,
    tokenizer: tiktoken.Encoding,
    prefixes: list[str],
    batch_size: int,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> list[str]:
    """Generates completions conditioned on prefixes

    Args:
        model: the language model
        device: device to put the tensors on
        tokenizer: the tokenizer
        prefixes: a list of strings as prefixes for generation
        batch_size: number of prefixes to batch together during generation
        max_new_tokens: the number of tokens to generate for each prefix
        temperature: temperature parameter of softmax

    Returns:
        a list of strings (continuations to prefixes)
    
    Note: you should implement a batched version of this function by
        left-padding tokenized prefixes with `tokenizer.eot_token` so that all
        sequences have equal length. `attention_mask` should be set to 0.0 for
        padding tokens, and 1.0 everywhere else.
    """
    model = model.to(device)
    model.eval()
    # Batch the prefixes:
    batches = [prefixes[i : i+batch_size] for i in range(0, len(prefixes), batch_size)]
    generations = []

    for batch in batches:
        tokenized = [tokenizer.encode(prefix) for prefix in batch]
        max_len = max([len(seq) for seq in tokenized])
        inputs = []
        masks = []
        for seq in tokenized:
            # Left-padding the sequence with tokenizer.eot_token
            input = [tokenizer.eot_token] * (max_len - len(seq)) + seq
            # Create the corresponding attention mask
            mask = [0.0] * (max_len - len(seq)) + [1.0] * len(seq)
            inputs.append(input)
            masks.append(mask)
        inputs = torch.tensor(inputs).to(device)
        masks = torch.tensor(masks).to(device)
        # Generate next tokens in an auto-regressive way
        for i in range(max_new_tokens):
            logits = model(inputs, masks)  # (B * S * V)
            # Taking all sequence in the batch, take only the last logit in each sequence,
            # and take the entire volcabulary for probabilities.
            last_logits = logits[:, -1, :]  # (B * V)
            probs = softmax_with_temperature(last_logits, temperature)  # (B * V)
            # torch.argmax(probs, dim=-1, keepdim=True) this approach is too deterministic,
            # a stochastic approach with multinomial sampling is used.
            next_token = torch.multinomial(probs, num_samples=1)
            # concatenate the new generated tokens to the inputs
            inputs = torch.cat([inputs, next_token], dim=-1)
            # update the masks by adding 1s
            masks = torch.cat([masks, torch.ones(next_token.shape).to(device)], dim=-1)

        for seq in inputs:
            generations.append(tokenizer.decode(seq.tolist()))

    return generations


def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=OmegaConf.load,
        required=True,
        help="the yaml config file used for model training",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        required=True,
        help="a json file with a list of strings as prefixes for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="temperature in sampling"
    )

    args = parser.parse_args()
    config = args.config
    with open(args.prefixes) as f:
        prefixes = [json.loads(line)["prefix"] for line in f]
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature

    # initialize tokenizer and model
    model_path = os.path.join(config.output_dir, "model.pt")
    assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    model.load_state_dict(torch.load(model_path))

    # generate and save outputs
    model.eval()
    generations = generate(
        model,
        device,
        tokenizer,
        prefixes,
        config.batch_size,
        max_new_tokens,
        temperature,
    )

    generation_path = os.path.join(config.output_dir, "generation.jsonl")
    print(f"writing generations to {generation_path}")
    with open(generation_path, "w") as f:
        for prefix, generation in zip(prefixes, generations):
            json.dump({"prefix": prefix, "generation": generation}, f)
            f.write("\n")

    print("done!")


if __name__ == "__main__":
    main()
