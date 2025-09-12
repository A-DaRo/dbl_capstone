# train.py

import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from datasets import load_dataset
from collections import defaultdict

# --- Import all project modules ---
from config import Config
from dataset import CoralscapesMTLDataset, TASK_DEFINITIONS
from augmentations import SegmentationAugmentation
from model import CoralMTLModel
from losses import CoralMTLLoss
from optimizer import create_optimizer_and_scheduler
from metrics import CoralMTLMetrics
from engine import train_one_epoch, validate_one_epoch
import reporting

def main():
    # --- 1. Setup and Configuration ---
    config = Config()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # --- 2. Data Preparation ---
    print("--- Loading and preparing datasets ---")
    hf_dataset = load_dataset("EPFL-ECEO/coralscapes")
    
    if config.DEBUG:
        print("!!! DEBUG MODE ENABLED: Using a small subset of the data. !!!")
        from datasets import DatasetDict
        train_subset = hf_dataset['train'].select(range(config.BATCH_SIZE * 2))
        val_subset = hf_dataset['validation'].select(range(config.BATCH_SIZE * 2))
        hf_dataset = DatasetDict({'train': train_subset, 'validation': val_subset})
        
    train_augs = SegmentationAugmentation(patch_size=config.PATCH_SIZE)
    train_dataset = CoralscapesMTLDataset(hf_dataset, split='train', augmentations=train_augs)
    val_dataset = CoralscapesMTLDataset(hf_dataset, split='validation', augmentations=None)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                              num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
                            
    # --- 3. Instantiate Model, Loss, Optimizer, Metrics ---
    num_classes = {task: len(info['id2label']) for task, info in TASK_DEFINITIONS.items()}
    
    print(f"--- Initializing Model ({config.ENCODER_NAME}) on {config.DEVICE} ---")
    model = CoralMTLModel(config.ENCODER_NAME, config.DECODER_CHANNEL, num_classes, config.ATTENTION_DIM).to(config.DEVICE)
    
    loss_fn = CoralMTLLoss(num_classes=num_classes, w_aux=config.W_AUX, w_consistency=config.W_CONSISTENCY,
                           hybrid_alpha=config.HYBRID_ALPHA, focal_gamma=config.FOCAL_GAMMA)
                           
    total_training_steps = len(train_loader) * config.NUM_EPOCHS
    warmup_steps = int(total_training_steps * config.WARMUP_STEPS_RATIO)
    
    optimizer, scheduler = create_optimizer_and_scheduler(model, learning_rate=config.LEARNING_RATE,
                                                          weight_decay=config.WEIGHT_DECAY,
                                                          adam_betas=config.ADAM_BETAS,
                                                          num_training_steps=total_training_steps,
                                                          num_warmup_steps=warmup_steps)
    
    scaler = torch.cuda.amp.GradScaler()
    metrics_calculator = CoralMTLMetrics(num_classes=num_classes, device=config.DEVICE)
    
    # --- 4. Training Loop ---
    best_h_mean = -1.0
    training_log_history = defaultdict(list)
    validation_log_history = defaultdict(list)

    print(f"--- Starting Training for {config.NUM_EPOCHS} epochs ---")
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{config.NUM_EPOCHS} =====")
        
        # Train one epoch
        train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, scaler, config.DEVICE, training_log_history)
        
        # Validate one epoch
        avg_val_loss = validate_one_epoch(model, val_loader, loss_fn, metrics_calculator, config.DEVICE)
        
        # Compute and log validation metrics for the epoch
        val_metrics = metrics_calculator.compute()
        for key, value in val_metrics.items():
            validation_log_history[key].append(value)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Avg Validation Loss: {avg_val_loss:.4f}")
        print(f"  Validation H-Mean: {val_metrics['H-Mean']:.4f} (Best: {max(best_h_mean, val_metrics['H-Mean']):.4f})")
        print(f"  Validation mIoU Genus: {val_metrics['mIoU_genus']:.4f}")
        print(f"  Validation mIoU Health: {val_metrics['mIoU_health']:.4f}")

        # Save the best model checkpoint based on H-Mean
        if val_metrics['H-Mean'] > best_h_mean:
            best_h_mean = val_metrics['H-Mean']
            save_path = os.path.join(config.OUTPUT_DIR, config.BEST_MODEL_NAME)
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New best model saved to {save_path}")

        # Reset metrics calculator for the next epoch
        metrics_calculator.reset()
        
    print("\n--- Training Complete ---")
    
    # --- 5. Post-Training Analysis and Plotting ---
    print("\n--- Generating final plots from training history ---")
    
    # Live Monitoring Plots
    reporting.plot_composite_losses(training_log_history, save_path=f"{config.OUTPUT_DIR}/composite_loss.png")
    reporting.plot_individual_task_losses(training_log_history, save_path=f"{config.OUTPUT_DIR}/individual_losses.png")
    reporting.plot_uncertainty_weights(training_log_history, save_path=f"{config.OUTPUT_DIR}/uncertainty.png")
    reporting.plot_learning_rate(training_log_history, warmup_steps, save_path=f"{config.OUTPUT_DIR}/lr_schedule.png")

    # Reporting Plots
    reporting.plot_primary_performance(validation_log_history, save_path=f"{config.OUTPUT_DIR}/primary_performance.png")

if __name__ == "__main__":
    main()