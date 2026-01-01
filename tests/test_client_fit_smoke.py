import pytest
from src.client import FlowerClient
from src.config import Config
import torch

def test_client_fit_smoke():
    """Test that client.fit runs for one epoch without error on toy data."""
    # Override config for speed
    Config.LOCAL_EPOCHS = 1
    Config.BATCH_SIZE = 2
    
    client = FlowerClient(data_path=None) # Use toy data
    
    # Get initial parameters
    params = client.get_parameters(config={})
    
    # Run fit
    updated_params, num_examples, metrics = client.fit(params, config={})
    
    assert len(updated_params) == len(params)
    assert num_examples > 0
    assert isinstance(metrics, dict)
