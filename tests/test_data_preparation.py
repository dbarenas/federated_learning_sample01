import os
import unittest
from src.data_preparation import extract_and_prepare_data


class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        """Set up a dummy PDF for testing."""
        self.test_data_dir = os.path.join("tests", "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)

        # We will use one of the generated PDFs for testing
        self.sample_pdf_path = os.path.join(
            "src", "data_generation", "factura_1.pdf"
        )
        if not os.path.exists(self.sample_pdf_path):
            # Generate it if it doesn't exist
            os.system("python3 src/data_generation/main.py")

        # Copy the sample PDF to the test data directory
        self.test_pdf_path = os.path.join(
            self.test_data_dir, "sample_invoice.pdf"
        )
        os.system(f"cp {self.sample_pdf_path} {self.test_pdf_path}")

    def test_extract_and_prepare_data(self):
        """Test the main data extraction and preparation function."""
        documents = extract_and_prepare_data(self.test_data_dir)

        self.assertIsInstance(documents, list)
        self.assertGreater(len(documents), 0, "No documents were processed.")

        doc = documents[0]
        self.assertIn("id", doc)
        self.assertIn("tokens", doc)
        self.assertIn("bboxes", doc)
        self.assertIn("ner_tags", doc)

        self.assertEqual(len(doc["tokens"]), len(doc["bboxes"]))
        self.assertEqual(len(doc["tokens"]), len(doc["ner_tags"]))

    def tearDown(self):
        """Clean up the test data directory."""
        if os.path.exists(self.test_data_dir):
            for file in os.listdir(self.test_data_dir):
                os.remove(os.path.join(self.test_data_dir, file))
            os.rmdir(self.test_data_dir)


if __name__ == "__main__":
    unittest.main()
