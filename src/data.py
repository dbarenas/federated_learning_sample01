import pandas as pd
from datasets import Dataset
import random
import os


def load_data(file_path=None):
    """
    Load dataset from a CSV file or generate a toy dataset.

    Expected CSV format: 'text', 'label' (0 or 1).
    """
    if file_path and os.path.exists(file_path):
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
    else:
        print("Using TOY dataset (no file provided or not found).")
        df = generate_toy_dataset()  # Renamed to avoid recursion

    # Ensure columns exist
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must have 'text' and 'label' columns.")

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Basic train/test split (80/20)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    return dataset


def generate_toy_dataset(num_samples=50):
    """Generate a dummy dataset for testing functionality."""
    sensitive_keywords = [
        "secret",
        "confidential",
        "password",
        "private",
        "salary",
        "ssn",
    ]
    normal_keywords = [
        "public",
        "weather",
        "recipe",
        "news",
        "hello",
        "meeting",
    ]

    data = []

    # Generate sensitive examples
    for _ in range(num_samples // 2):
        text = (
            f"This is a {random.choice(sensitive_keywords)} document with "
            f"id {random.randint(1000, 9999)}."
        )
        data.append({"text": text, "label": 1})

    # Generate non-sensitive examples
    for _ in range(num_samples // 2):
        text = (
            f"This is a {random.choice(normal_keywords)} update about "
            f"{random.randint(1, 100)}."
        )
        data.append({"text": text, "label": 0})

    random.shuffle(data)
    return pd.DataFrame(data)
