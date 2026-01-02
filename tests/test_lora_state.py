import numpy as np
from src.common import load_model, get_parameters, set_parameters


def test_lora_model_creation():
    """Verify that the model has trainable LoRA parameters."""
    model, _ = load_model()
    params = get_parameters(model)
    assert len(params) > 0, "Model should have some trainable parameters."
    assert all(
        "lora" in name or "classifier" in name
        for name, _ in model.named_parameters()
        if _.requires_grad
    ), "Only LoRA and classifier parameters should be trainable."


def test_parameter_setting_and_getting():
    """Test if setting parameters changes the model's state as expected."""
    model, _ = load_model()
    initial_params = get_parameters(model)

    # Create a new set of parameters with different values
    new_params = [
        np.random.randn(*p.shape).astype(p.dtype) for p in initial_params
    ]

    # Set the new parameters
    set_parameters(model, new_params)
    retrieved_params = get_parameters(model)

    # Verify that the parameters were updated
    for i in range(len(new_params)):
        assert np.allclose(
            new_params[i], retrieved_params[i]
        ), f"Parameter {i} was not updated correctly."
