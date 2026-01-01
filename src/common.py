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
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    
    # We only want to load the keys that are present in the state_dict matching the model
    # FL usually sends all parameters if we sent all parameters.
    # But here we should be careful to only update trainable ones if we filtered earlier.
    # simpler approach: strict=True if we are consistent.
    
    # However, in this simple implementation, we assume we pass EVERYTHING that we extracted.
    model.load_state_dict(state_dict, strict=True)

def get_parameters(model) -> List[np.ndarray]:
    """Get the parameters of the model as a list of numpy arrays (for Flower server)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
