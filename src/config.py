import os

class Config:
    # Model settings
    MODEL_NAME = "distilbert-base-multilingual-cased"
    MAX_LENGTH = 128
    NUM_LABELS = 2
    
    # LoRA settings
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    TARGET_MODULES = ["q_lin", "v_lin"]
    
    # Federated Learning settings
    SERVER_ADDRESS = "127.0.0.1:8080"
    NUM_ROUNDS = 3
    MIN_FIT_CLIENTS = 2
    MIN_AVAILABLE_CLIENTS = 2
    
    # Training settings
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-4
    
    # Paths
    DATA_PATH = "data.csv" # If using real data
    MODEL_SAVE_DIR = "saved_models"

    # Inference
    SENSITIVITY_THRESHOLD = 0.5
