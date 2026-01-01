from src.common import load_model, Config

def test_tokenization_shapes():
    _, tokenizer = load_model()
    text = "This is a test sentence."
    
    inputs = tokenizer(
        text, 
        truncation=True, 
        padding="max_length", 
        max_length=Config.MAX_LENGTH, 
        return_tensors="pt"
    )
    
    assert inputs["input_ids"].shape == (1, Config.MAX_LENGTH)
    assert inputs["attention_mask"].shape == (1, Config.MAX_LENGTH)
