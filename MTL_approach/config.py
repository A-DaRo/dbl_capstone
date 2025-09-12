# config.py

import torch

class Config:
    """A single class to hold all training configuration."""
    # --- Dataset and DataLoader ---
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PATCH_SIZE = 512

    # --- Model Architecture ---
    ENCODER_NAME = "nvidia/mit-b2"
    DECODER_CHANNEL = 256
    ATTENTION_DIM = 128

    # --- Optimizer and Scheduler ---
    LEARNING_RATE = 6e-5
    WEIGHT_DECAY = 0.01
    ADAM_BETAS = (0.9, 0.999)
    NUM_EPOCHS = 100
    WARMUP_STEPS_RATIO = 0.1  # 10% of total steps for warmup

    # --- Loss Function ---
    W_AUX = 0.4
    W_CONSISTENCY = 0.1
    HYBRID_ALPHA = 0.5
    FOCAL_GAMMA = 2.0

    # --- Training ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    OUTPUT_DIR = "training_outputs"
    BEST_MODEL_NAME = "best_model.pth"
    
    # --- Debugging ---
    # If True, uses a tiny subset of the data for a quick test run of the pipeline.
    DEBUG = Falses