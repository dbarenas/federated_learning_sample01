import torch
from transformers import AutoTokenizer, LayoutLMForTokenClassification
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from collections import OrderedDict
from typing import List, Dict, Any
import numpy as np
import json
from pathlib import Path
import importlib.metadata
from datetime import datetime

from .config import Config


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_base_model_and_tokenizer():
    """Load the base model and tokenizer without PEFT wrapping."""
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = LayoutLMForTokenClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=Config.NUM_LABELS
    )
    return model, tokenizer


def load_model():
    """Load the base model and tokenizer, then apply LoRA."""
    model, tokenizer = load_base_model_and_tokenizer()

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        target_modules=Config.TARGET_MODULES,
    )

    # Wrap model with PEFT
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def get_lora_parameters(model) -> List[np.ndarray]:
    """Get the LoRA model parameters as a list of numpy arrays."""
    state_dict = get_lora_state_dict(model)
    return [val.cpu().numpy() for val in state_dict.values()]


def set_lora_parameters(model, parameters: List[np.ndarray]) -> None:
    """Set the LoRA model parameters from a list of numpy arrays."""
    state_dict = get_lora_state_dict(model)
    params_dict = zip(state_dict.keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    set_lora_state_dict(model, state_dict)


def get_lora_state_dict(model) -> Dict[str, torch.Tensor]:
    """Get the LoRA state dictionary with sorted keys."""
    state_dict = model.state_dict()
    return OrderedDict(
        (k, v) for k, v in sorted(state_dict.items()) if "lora_" in k
    )


def set_lora_state_dict(model, state_dict: Dict[str, torch.Tensor]) -> None:
    """Set the LoRA state dictionary."""
    model.load_state_dict(state_dict, strict=False)


# New artifact utility functions
def ensure_artifact_dir(base_dir: str | Path, round_num: int) -> Path:
    """Ensure the artifact directory for a given round exists and returns its path."""
    base_dir = Path(base_dir)
    round_dir = base_dir / f"round_{round_num:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    return round_dir


def save_lora_artifact(
    model, tokenizer, out_dir: Path, metadata: Dict[str, Any]
) -> None:
    """Save LoRA artifact (adapters, config, tokenizer, metadata)."""
    # 1. Save LoRA adapter and config
    model.save_pretrained(out_dir)

    # 2. Save tokenizer
    tokenizer_dir = out_dir / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)

    # 3. Save metadata
    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts saved to {out_dir}")


def load_lora_artifact(
    model, artifact_dir: Path
) -> tuple[Any, AutoTokenizer, Dict[str, Any]]:
    """Load LoRA artifact and apply it to a base model."""
    artifact_dir = Path(artifact_dir)

    # 1. Load adapter - convert Path to string for PeftModel
    model = PeftModel.from_pretrained(model, str(artifact_dir))

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir / "tokenizer"))

    # 3. Load metadata
    metadata_path = artifact_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print(f"Loaded LoRA artifact from: {artifact_dir}")
    return model, tokenizer, metadata


def get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    packages = ["torch", "transformers", "peft", "flwr"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "Not Found"
    return versions
