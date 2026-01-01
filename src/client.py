import argparse
import warnings
from collections import OrderedDict

import flwr as fl
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from .common import load_model, set_parameters, get_parameters, get_device
from .data import load_data
from .config import Config

warnings.filterwarnings("ignore", category=UserWarning)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, data_path=None):
        self.device = get_device()
        self.model, self.tokenizer = load_model()
        self.model.to(self.device)
        
        # Load Data
        dataset = load_data(data_path)
        self.train_dataset = dataset["train"]
        self.test_dataset = dataset["test"]

        # Tokenization function
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding=False, max_length=Config.MAX_LENGTH)

        # Preprocess datasets
        self.train_dataset = self.train_dataset.map(tokenize_function, batched=True)
        self.test_dataset = self.test_dataset.map(tokenize_function, batched=True)
        
        # Format for PyTorch
        self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        self.test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def get_parameters(self, config):
        """Return current model weights."""
        return get_parameters(self.model)

    def fit(self, parameters, config):
        """Train the model using the parameters from the server."""
        set_parameters(self.model, parameters)
        
        # Training Loop
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True, 
            collate_fn=self.data_collator
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
                
        return get_parameters(self.model), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model using parameters from the server."""
        set_parameters(self.model, parameters)
        
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=Config.BATCH_SIZE, 
            collate_fn=self.data_collator
        )
        
        self.model.eval()
        loss = 0.0
        preds = []
        labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss += outputs.loss.item()
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                preds.extend(predictions.cpu().numpy())
                labels.extend(batch["label"].cpu().numpy())
                
        accuracy = accuracy_score(labels, preds)
        return float(loss / len(test_loader)), len(self.test_dataset), {"accuracy": float(accuracy)}

def main():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--client-id", type=str, default="1", help="Client ID")
    parser.add_argument("--server-address", type=str, default=Config.SERVER_ADDRESS, help="Server Address")
    parser.add_argument("--data-path", type=str, default=None, help="Path to CSV dataset")
    args = parser.parse_args()
    
    print(f"Starting Client {args.client_id} connecting to {args.server_address}")
    
    client = FlowerClient(data_path=args.data_path)
    
    fl.client.start_numpy_client(
        server_address=args.server_address, 
        client=client
    )

if __name__ == "__main__":
    main()
