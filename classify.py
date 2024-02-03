import argparse
import os
import tiktoken
import torch
from datasets import load_dataset
from model import DecoderLM
from omegaconf import OmegaConf
from tqdm import trange
from utils import determine_device, enable_tf32

YELP_TEMPLATE = "Here is a yelp review.\n{text}\nThis review is"
YELP_LABEL_MAP = {0: " negative", 1: " positive"}


@torch.inference_mode()
def score(
    model: DecoderLM,
    device: str,
    tokenizer: tiktoken.Encoding,
    texts: list[str],
    batch_size: int,
) -> torch.FloatTensor:
    """Scores all possible next tokens for the given texts

    Args:
        model: the language model
        device: device to put the tensors on
        tokenizer: the tokenizer
        texts: a list of strings for scoring
        batch_size: number of instances to score during one forward pass

    Returns:
        Logits corresponding to next token probabilities (B x V).

    
    Note: you should implement a batched version of this function by
        left-padding tokenized instances with `tokenizer.eot_token` so that all
        sequences have equal length. `attention_mask` should be set to 0.0 for
        padding tokens, and 1.0 everywhere else.
    """
    # Tokenize the texts
    tokenized = [tokenizer.encode(text) for text in texts]
    # Compute the max length for padding
    max_len = max([len(seq) for seq in tokenized])

    inputs_batches = []
    masks = []
    for seq in tokenized:
        # Left-padding the sequences
        padded = [tokenizer.eot_token] * (max_len - len(seq)) + seq
        # Create the corresponding masks
        mask = [0.0] * (max_len - len(seq)) + [1.0] * len(seq)
        inputs_batches.append(padded)
        masks.append(mask)
    inputs_batches = torch.tensor(inputs_batches).to(device)
    masks = torch.tensor(masks).to(device)

    logit_list = []
    model.eval()  # model set to evaluation mode
    for i in range(0, len(texts), batch_size):
        batch = inputs_batches[i : i+batch_size]
        attention_mask = masks[i : i+batch_size]
        # Disable gradient calculations.
        with torch.no_grad():
            out = model(batch, attention_mask)
        out = out[:, -1, :]
        # Move the logits to cpu. Otherwise, there will be the CUDA out of memory issue.
        out_cpu = out.cpu() 
        logit_list.append(out_cpu)
    logits = torch.cat(logit_list, dim=0)  # (B * V)

    return logits


def classify_binary_sentiment(
    logits: torch.FloatTensor,
    tokens_of_interest: list[int],
    calibrate: bool = False,
) -> list[int]:
    """
    Args:
        logits: torch tensor corresponding to next token probabilities (B x V)
        tokens_of_interest: the indices for the tokens corresponding to negative
          and positive labels
        calibrate: when calibration is true, set the threshold according to your
          proposed calibration strategy in Question 3.6
    Returns:
        A list of predictions with length B, an element should be 0 if the
          negative class is more likely and 1 if the positive class is more
          likely.
    """

    probs = logits[:, tokens_of_interest].softmax(1)

    if calibrate:
        threshold = probs[:, 1].median().item()  # Take the median of all probabilities for the positive class ([:, 1])
    else:
        threshold = 0.5
    
    predictions = (probs[:, 1] > threshold).int().tolist()
    return predictions


def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=OmegaConf.load,
        required=True,
        help="the yaml config file used for model training",
    )
    parser.add_argument("--calibrate", action="store_true")

    args = parser.parse_args()
    config = args.config

    # initialize tokenizer and model
    model_path = os.path.join(config.output_dir, "model.pt")
    assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    model.load_state_dict(torch.load(model_path))

    dataset = load_dataset("yelp_polarity")
    test_subset = (
        dataset["test"]
        .filter(
            lambda instance: len(
                tokenizer.encode(YELP_TEMPLATE.format(text=instance["text"]))
            )
            <= model.n_positions
        )
        .shuffle(seed=42)[:1000]
    )
    texts = [YELP_TEMPLATE.format(text=text) for text in test_subset["text"]]
    negative_token_id = tokenizer.encode_single_token(YELP_LABEL_MAP[0])
    positive_token_id = tokenizer.encode_single_token(YELP_LABEL_MAP[1])

    model.eval()
    logits = score(
        model,
        device,
        tokenizer,
        texts,
        config.batch_size,
    )

    predictions = classify_binary_sentiment(
        logits, [negative_token_id, positive_token_id], calibrate=args.calibrate
    )

    acc = sum(
        1 if pred == label else 0
        for pred, label in zip(predictions, test_subset["label"])
    ) / len(predictions)
    print(f"accuracy on yelp: {acc * 100:.1f}")


if __name__ == "__main__":
    main()
