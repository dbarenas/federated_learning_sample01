import pytest
import numpy as np
from src.common import load_model, get_parameters, set_parameters

def test_lora_model_creation():
    """Test that the model is created with LoRA adapters."""
    model, tokenizer = load_model()
    # Check if we have peft attributes
    assert hasattr(model, "peft_config")
    # Check if number of parameters is less than full model (efficient)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    assert trainable_params < all_params
    assert trainable_params > 0

def test_parameter_exchange():
    """Test that we can extract and set parameters without error."""
    model, _ = load_model()
    
    # Get initial params
    initial_params = get_parameters(model)
    initial_params_copy = [p.copy() for p in initial_params]
    assert len(initial_params) > 0
    assert isinstance(initial_params[0], np.ndarray)
    
    # Perturb parameters slightly to simulate update
    new_params = [p + 1.0 for p in initial_params]
    
    # Set parameters
    set_parameters(model, new_params)
    
    # Check if updated
    current_params = get_parameters(model)
    assert np.allclose(current_params[0], new_params[0])
    assert not np.allclose(current_params[0], initial_params_copy[0])
