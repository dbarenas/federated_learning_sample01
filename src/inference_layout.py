import argparse
import torch
import os

from .common import get_device, load_model
from .data_preparation import extract_and_prepare_data, id2label
from .config import Config


def run_inference(model, tokenizer, pdf_path: str, device):
    """
    Runs inference on a single PDF file and prints the predicted entities.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at '{pdf_path}'")
        return

    # Process the single PDF
    pdf_dir = os.path.dirname(pdf_path)
    file_name = os.path.basename(pdf_path)

    all_docs = extract_and_prepare_data(pdf_dir)
    doc_to_process = next(
        (doc for doc in all_docs if doc["id"] == file_name), None
    )

    if not doc_to_process:
        print(f"Could not process the PDF file: {pdf_path}")
        return

    # Prepare inputs for the model
    tokens = doc_to_process["tokens"]
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=Config.MAX_LENGTH,
    )
    encoding["bbox"] = torch.tensor([doc_to_process["bboxes"]]).to(device)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # Run model inference
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)

    # Get predictions
    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
    word_ids = encoding.word_ids(batch_index=0)

    # Print extracted entities
    print(f"Extracted entities from '{os.path.basename(pdf_path)}':")
    previous_word_idx = None
    for idx, pred in enumerate(predictions):
        if word_ids[idx] is None or word_ids[idx] == previous_word_idx:
            continue

        label = id2label[pred]
        if label != "O":
            token_text = tokens[word_ids[idx]]
            print(f"  - {label}: {token_text}")

        previous_word_idx = word_ids[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for LayoutLM model on invoices."
    )
    parser.add_argument(
        "--pdf-path",
        type=str,
        required=True,
        help="Path to the PDF invoice file to process.",
    )
    args = parser.parse_args()

    device = get_device()

    # Load the base model and tokenizer
    model, tokenizer = load_model()
    model.to(device)

    print("Running inference...")
    run_inference(model, tokenizer, args.pdf_path, device)


if __name__ == "__main__":
    main()
