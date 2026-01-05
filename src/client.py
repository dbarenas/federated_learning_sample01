import argparse
import warnings
import os
from datasets import Dataset

import flwr as fl
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import DataCollatorForTokenClassification
from tqdm import tqdm

from .common import (
    load_model,
    set_lora_parameters,
    get_lora_parameters,
    get_device,
)
from .data_preparation import extract_and_prepare_data
from .config import Config

warnings.filterwarnings("ignore", category=UserWarning)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, data_path=None):
        self.device = get_device()
        self.model, self.tokenizer = load_model()
        self.model.to(self.device)

        # Load and Prepare Data
        if data_path is None:
            data_path = os.path.join(
                os.path.dirname(__file__), "data_generation"
            )
            if not os.path.exists(data_path) or not os.listdir(data_path):
                print("Toy data not found, generating it now...")
                os.system("python3 src/data_generation/main.py")

        print(f"Loading data from {data_path}...")
        documents = extract_and_prepare_data(data_path)
        if not documents:
            raise ValueError(
                f"No documents found in {data_path}. "
                "Run data generation first."
            )
        dataset = Dataset.from_list(documents)

        # Split dataset
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        self.train_dataset = dataset["train"]
        self.test_dataset = dataset["test"]

        # Tokenization function for LayoutLM
        def tokenize_and_align(examples):
            encoding = self.tokenizer(
                examples["tokens"],
                is_split_into_words=True,
                padding="max_length",
                truncation=True,
                max_length=Config.MAX_LENGTH,
            )

            labels = []
            bboxes = []
            for i, label_list in enumerate(examples["ner_tags"]):
                word_ids = encoding.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                bbox_list = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                        bbox_list.append([0, 0, 0, 0])
                    elif word_idx != previous_word_idx:
                        label_ids.append(label_list[word_idx])
                        bbox_list.append(examples["bboxes"][i][word_idx])
                    else:
                        label_ids.append(-100)
                        bbox_list.append(examples["bboxes"][i][word_idx])
                    previous_word_idx = word_idx
                labels.append(label_ids)
                bboxes.append(bbox_list)

            encoding["labels"] = labels
            encoding["bbox"] = bboxes
            return encoding

        # Preprocess datasets
        self.train_dataset = self.train_dataset.map(
            tokenize_and_align,
            batched=True,
            remove_columns=self.train_dataset.column_names,
        )
        self.test_dataset = self.test_dataset.map(
            tokenize_and_align,
            batched=True,
            remove_columns=self.test_dataset.column_names,
        )

        # Format for PyTorch
        self.train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "bbox", "labels"],
        )
        self.test_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "bbox", "labels"],
        )

        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer
        )

    def get_parameters(self, config):
        """Return current model weights."""
        return get_lora_parameters(self.model)

    def fit(self, parameters, config):
        """Train the model using the parameters from the server."""
        set_lora_parameters(self.model, parameters)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            collate_fn=self.data_collator,
        )

        optimizer = AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.model.train()

        for epoch in range(Config.LOCAL_EPOCHS):
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return get_lora_parameters(self.model), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model using parameters from the server."""
        set_lora_parameters(self.model, parameters)

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=Config.BATCH_SIZE,
            collate_fn=self.data_collator,
        )

        self.model.eval()
        loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss += outputs.loss.item()

        # Placeholder for a more sophisticated metric (e.g., seqeval)
        return (
            float(loss / len(test_loader)),
            len(self.test_dataset),
            {"accuracy": 0.0},
        )


def main():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument(
        "--client-id", type=str, default="1", help="Client ID"
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=Config.SERVER_ADDRESS,
        help="Server Address",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to directory with PDF invoices.",
    )
    args = parser.parse_args()

    print(
        f"Starting Client {args.client_id} "
        f"connecting to {args.server_address}"
    )

    client = FlowerClient(data_path=args.data_path)

    fl.client.start_numpy_client(
        server_address=args.server_address, client=client
    )


if __name__ == "__main__":
    main()
