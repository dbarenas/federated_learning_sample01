import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import os
from typing import Optional

from .common import load_base_model_and_tokenizer, get_device, load_lora_artifact
from .config import Config


def find_latest_artifact(artifacts_dir: Path) -> Optional[Path]:
    """Find the path to the latest round's artifact."""
    if not artifacts_dir.exists():
        return None

    round_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name.startswith("round_")]
    if not round_dirs:
        return None

    # Sort by round number (e.g., round_003)
    latest_round_dir = max(round_dirs, key=lambda d: int(d.name.split('_')[-1]))
    return latest_round_dir


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

    # Note: This is a placeholder for a more complex prediction logic
    # that would extract meaningful information from the tokens.
    # Here, we just return the probability of the first token being "sensitive".
    score_sensitive = probs[0][1][1].item()
    is_sensitive = score_sensitive >= Config.SENSITIVITY_THRESHOLD

    return is_sensitive, score_sensitive


def main():
    parser = argparse.ArgumentParser(description="Inference Script")
    parser.add_argument(
        "--text", type=str, required=True, help="Text to classify"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=os.environ.get("ARTIFACTS_DIR", "./artifacts"),
        help="Directory where artifacts are saved.",
    )
    args = parser.parse_args()

    device = get_device()

    # Load base model
    model, tokenizer = load_base_model_and_tokenizer()

    # Find and load the latest LoRA artifact if available
    artifacts_dir = Path(args.artifacts_dir)
    latest_artifact_path = find_latest_artifact(artifacts_dir)

    if latest_artifact_path:
        print(f"Found artifact at: {latest_artifact_path}")
        try:
            model, tokenizer, metadata = load_lora_artifact(model, latest_artifact_path)
            print("Successfully loaded LoRA adapter and tokenizer from artifact.")
            print(f"  > Trained for {metadata.get('round')} rounds")
            print(f"  > Timestamp: {metadata.get('timestamp')}")
        except Exception as e:
            print(f"Error loading artifact: {e}")
            print("Fell back to using the base model without LoRA weights.")
    else:
        print("No artifact found.")
        print("Using the base model without LoRA weights.")

    model.to(device)
    model.eval()

    is_sensitive, score = predict(args.text, model, tokenizer, device)

    print("-" * 30)
    print(f"Text: {args.text}")
    print(f"Sensitive Probability: {score:.4f}")
    print(f"Prediction: {'SENSITIVE' if is_sensitive else 'NOT SENSITIVE'}")
    print("-" * 30)


if __name__ == "__main__":
    main()
