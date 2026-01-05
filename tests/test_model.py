import unittest
import torch
from src.common import load_model, get_device
from src.config import Config


class TestModel(unittest.TestCase):
    def test_load_model(self):
        """Test that the LayoutLM model is loaded correctly."""
        model, tokenizer = load_model()

        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)

        # Check if the model is a LayoutLM model for token classification
        self.assertEqual(model.config.model_type, "layoutlm")
        self.assertIn(
            "LayoutLMForTokenClassification", str(type(model.base_model.model))
        )

    def test_model_forward_pass(self):
        """Test a forward pass of the model with dummy data."""
        device = get_device()
        model, _ = load_model()
        model.to(device)

        # Create a dummy input batch
        batch_size = 2
        seq_length = 128
        vocab_size = 30522  # Vocabulary size for layoutlm-base-uncased

        # Create realistic bounding boxes
        x0 = torch.randint(0, 400, (batch_size, seq_length))
        y0 = torch.randint(0, 400, (batch_size, seq_length))
        x1 = x0 + torch.randint(0, 100, (batch_size, seq_length))
        y1 = y0 + torch.randint(0, 100, (batch_size, seq_length))
        bbox = torch.stack([x0, y0, x1, y1], dim=-1).to(device)

        dummy_input = {
            "input_ids": torch.randint(
                0, vocab_size, (batch_size, seq_length)
            ).to(device),
            "attention_mask": torch.ones(
                batch_size, seq_length, dtype=torch.int64
            ).to(device),
            "bbox": bbox,
            "labels": torch.randint(
                0, Config.NUM_LABELS, (batch_size, seq_length)
            ).to(device),
        }

        # Run a forward pass
        with torch.no_grad():
            outputs = model(**dummy_input)

        # Check the output shape
        self.assertIn("loss", outputs)
        self.assertIn("logits", outputs)
        self.assertEqual(
            outputs.logits.shape, (batch_size, seq_length, Config.NUM_LABELS)
        )


if __name__ == "__main__":
    unittest.main()
