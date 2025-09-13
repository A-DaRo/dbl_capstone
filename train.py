import torch
import numpy as np
import os
import math
from torch.utils.data import DataLoader
from datasets import load_dataset
from collections import defaultdict

# --- Import all project modules from the new 'src' structure ---
# NOTE: Assumes 'src' is in PYTHONPATH or the package is installed.
from coral_mtl.data.dataset import CoralscapesMTLDataset, TASK_DEFINITIONS
from coral_mtl.data.augmentations import SegmentationAugmentation
from coral_mtl.model.core import CoralMTLModel
from coral_mtl.engine.losses import CoralMTLLoss
from coral_mtl.engine.optimizer import create_optimizer_and_scheduler
from coral_mtl.engine.metrics import CoralMTLMetrics
from coral_mtl.engine.trainer import Trainer
from coral_mtl.utils.visualization import (
    plot_composite_losses, 
    plot_individual_task_losses, 
    plot_uncertainty_weights, 
    plot_learning_rate, 
    plot_primary_performance
)
# A simple config class for demonstration; this would be replaced by YAML loading.
from config import Config 

def main():
    # --- 1. Setup and Configuration ---
    config = Config()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # --- 2. Data Preparation ---
    print("--- Loading and preparing datasets ---")
    hf_dataset = load_dataset("EPFL-ECEO/coralscapes")
    
    train_augs = SegmentationAugmentation(patch_size=config.PATCH_SIZE)
    train_dataset = CoralscapesMTLDataset(hf_dataset, split='train', augmentations=train_augs, patch_size=config.PATCH_SIZE)
    val_dataset = CoralscapesMTLDataset(hf_dataset, split='validation', augmentations=None, patch_size=config.PATCH_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                              num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
                            
    # --- 3. Instantiate Model, Loss, Optimizer, Metrics for MTL ---
    num_classes = {task: len(info['id2label']) for task, info in TASK_DEFINITIONS.items()}
    
    model = CoralMTLModel(config.ENCODER_NAME, config.DECODER_CHANNEL, num_classes, config.ATTENTION_DIM).to(config.DEVICE)
    loss_fn = CoralMTLLoss(num_classes=num_classes, ignore_index=0).to(config.DEVICE)
                           
    total_steps = math.ceil(len(train_loader) / config.GRADIENT_ACCUMULATION_STEPS) * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_STEPS_RATIO)
    
    optimizer, scheduler = create_optimizer_and_scheduler(model, learning_rate=config.LEARNING_RATE,
                                                          weight_decay=config.WEIGHT_DECAY,
                                                          num_training_steps=total_steps,
                                                          num_warmup_steps=warmup_steps)
    
    metrics_calculator = CoralMTLMetrics(num_classes=num_classes, device=config.DEVICE)
    
    # --- 4. Instantiate and Run the Trainer ---
    trainer = Trainer(model, train_loader, val_loader, loss_fn, metrics_calculator,
                      optimizer, scheduler, config)
    
    _, training_log, validation_log = trainer.train()
    
    # --- 5. Post-Training Analysis and Plotting ---
    print("\n--- Generating final plots from training history ---")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plot_composite_losses(training_log, save_path=f"{config.OUTPUT_DIR}/composite_loss.png")
    plot_individual_task_losses(training_log, save_path=f"{config.OUTPUT_DIR}/individual_losses.png")
    plot_uncertainty_weights(training_log, save_path=f"{config.OUTPUT_DIR}/uncertainty.png")
    plot_learning_rate(training_log, warmup_steps, save_path=f"{config.OUTPUT_DIR}/lr_schedule.png")
    plot_primary_performance(validation_log, save_path=f"{config.OUTPUT_DIR}/primary_performance.png")

if __name__ == "__main__":
    main()