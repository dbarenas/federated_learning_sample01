import pdfplumber
import os
from typing import List, Dict, Any
import re

# Define the entity labels
LABELS = [
    "O",
    "B-TOTAL",
    "I-TOTAL",
    "B-DATE",
    "I-DATE",
    "B-INVOICE_ID",
    "I-INVOICE_ID",
]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}


def normalize_bbox(bbox, width, height):
    """Normalize bounding box coordinates to a 0-1000 scale."""
    return [
        int(1000 * (bbox["x0"] / width)),
        int(1000 * (bbox["top"] / height)),
        int(1000 * (bbox["x1"] / width)),
        int(1000 * (bbox["bottom"] / height)),
    ]


def label_invoice_data(
    words: List[Dict[str, Any]], width: float, height: float
) -> Dict[str, Any]:
    """
    Applies simple rule-based labeling to invoice words and formats for
    LayoutLM.
    """
    tokens = [word["text"] for word in words]
    bboxes = [normalize_bbox(word, width, height) for word in words]

    # Initialize labels with "O" (Outside)
    ner_tags = [label2id["O"]] * len(tokens)

    for i, token in enumerate(tokens):
        # Rule for Invoice ID (e.g., "F-2025-00001")
        if re.match(r"^F-\d{4}-\d{5}$", token):
            ner_tags[i] = label2id["B-INVOICE_ID"]

        # Rule for Date (e.g., "dd/mm/yyyy")
        if re.match(r"^\d{2}/\d{2}/\d{4}$", token):
            ner_tags[i] = label2id["B-DATE"]

        # Rule for Total Amount
        if token.upper() in ["TOTAL", "FACTURA:"]:
            line_words = [
                (j, w)
                for j, w in enumerate(words)
                if abs(w["top"] - words[i]["top"]) < 5
            ]

            for j, w in reversed(line_words):
                if re.match(r"^\d{1,3}(?:\.\d{3})*,\d{2}\s*€?$", w["text"]):
                    ner_tags[j] = label2id["B-TOTAL"]
                    if j > 0 and re.match(
                        r"^\d{1,3}(?:\.\d{3})*,\d{2}$", tokens[j - 1]
                    ):
                        ner_tags[j - 1] = label2id["I-TOTAL"]
                    break

    return {
        "tokens": tokens,
        "bboxes": bboxes,
        "ner_tags": ner_tags,
    }


def extract_and_prepare_data(pdf_directory: str) -> List[Dict[str, Any]]:
    """
    Extracts and prepares data from all PDF files in a directory for a
    LayoutLM model.
    """
    prepared_documents = []
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_directory, pdf_file)

        try:
            with pdfplumber.open(file_path) as pdf:
                if not pdf.pages:
                    continue
                page = pdf.pages[0]

                words = page.extract_words(use_text_flow=True)

                if not words:
                    continue

                document_width = float(page.width)
                document_height = float(page.height)

                labeled_data = label_invoice_data(
                    words, document_width, document_height
                )
                labeled_data["id"] = os.path.basename(file_path)

                prepared_documents.append(labeled_data)

        except Exception as e:
            print(f"Could not process file {file_path}: {e}")

    return prepared_documents


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data_generation")

    if not os.path.exists(data_dir):
        print(f"Error: Directory not found at '{data_dir}'")
        print("Please run 'src/data_generation/main.py' first.")
    else:
        documents = extract_and_prepare_data(data_dir)

        if documents:
            print(f"Successfully processed and labeled {len(documents)} docs.")

            first_doc = documents[0]
            print(f"\nSample data from: {first_doc['id']}")
            print(f"Found {len(first_doc['tokens'])} tokens.")

            print("\nSample of labeled tokens:")
            for token, bbox, tag_id in zip(
                first_doc["tokens"][:15],
                first_doc["bboxes"][:15],
                first_doc["ner_tags"][:15],
            ):
                print(
                    f"  - Token: '{token}', Bbox: {bbox}, "
                    f"Tag: {id2label[tag_id]}"
                )

            print("\nVerification of 'TOTAL' labeling:")
            found_total = False
            for token, tag_id in zip(
                first_doc["tokens"], first_doc["ner_tags"]
            ):
                if id2label[tag_id] != "O":
                    print(f"  - Token: '{token}', Tag: {id2label[tag_id]}")
                    found_total = True
            if not found_total:
                print("  - No specific tags found in this sample.")

        else:
            print(f"No PDF documents could be processed in '{data_dir}'.")
