import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from collections import OrderedDict
from typing import List, Dict
import numpy as np

from .config import Config

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load the base model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Load model with classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, 
        num_labels=Config.NUM_LABELS
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        target_modules=Config.TARGET_MODULES
    )
    
    # Wrap model with PEFT
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def set_parameters(model, parameters: List[np.ndarray]):
    """Set the parameters of the model from a list of numpy arrays (from Flower server).
    Only sets the LoRA parameters + classifier head if trainable.
    """
    trainable_keys = [
        name for name, _ in model.state_dict().items() if "lora" in name or "classifier" in name
    ]
    params_dict = zip(trainable_keys, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)

def get_parameters(model) -> List[np.ndarray]:
    """Get the parameters of the model as a list of numpy arrays (for Flower server)."""
    return [
        val.cpu().numpy()
        for name, val in model.state_dict().items()
        if "lora" in name or "classifier" in name
    ]
