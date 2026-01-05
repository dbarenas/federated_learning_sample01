import argparse
import torch
import torch.nn.functional as F
from .common import load_model, get_device
from .config import Config


def predict(text, model, tokenizer, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=Config.MAX_LENGTH,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)

    score_sensitive = probs[0][1].item()
    is_sensitive = score_sensitive >= Config.SENSITIVITY_THRESHOLD

    return is_sensitive, score_sensitive


def main():
    parser = argparse.ArgumentParser(description="Inference Script")
    parser.add_argument(
        "--text", type=str, required=True, help="Text to classify"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to loaded PEFT model (optional, uses default otherwise)",
    )
    args = parser.parse_args()

    device = get_device()
    print("Loading model...")
    model, tokenizer = load_model()
    model.to(device)
    model.eval()

    # Note: If we had a saved adapter, we would load it here.
    # For this simple script, it uses the initialized (untrained or default)
    # weights unless we add logic to load specific saved weights.

    is_sensitive, score = predict(args.text, model, tokenizer, device)

    print("-" * 30)
    print(f"Text: {args.text}")
    print(f"Sensitive Probability: {score:.4f}")
    print(f"Prediction: {'SENSITIVE' if is_sensitive else 'NOT SENSITIVE'}")
    print("-" * 30)


if __name__ == "__main__":
    main()
