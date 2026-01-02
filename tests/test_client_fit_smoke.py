from src.client import FlowerClient
from src.config import Config


def test_client_fit_smoke():
    """Test that client.fit runs for one epoch without error on toy data."""
    # Override config for speed
    Config.LOCAL_EPOCHS = 1
    Config.BATCH_SIZE = 2

    try:
        client = FlowerClient(data_path=None)  # Use toy data
    except ValueError as e:
        if "No documents found" in str(e):
            # This can happen if the toy data generation fails.
            # We'll generate the data and try again.
            import os
            os.system("python3 src/data_generation/main.py")
            client = FlowerClient(data_path=None)
        else:
            raise e


    # Get initial parameters
    params = client.get_parameters(config={})

    # Run fit
    updated_params, num_examples, metrics = client.fit(params, config={})

    assert len(updated_params) == len(params)
    assert num_examples > 0
    assert isinstance(metrics, dict)
