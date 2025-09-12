# File: tune_baseline_advanced.py
# Description: An advanced Optuna-based hyperparameter tuning script for the
#              single-task SegFormer baseline. This version explores a much
#              wider search space, including different loss functions and stronger
#              augmentations, with an increased batch size and no gradient accumulation.

import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import numpy as np
import os
import math
from collections import defaultdict
from typing import Dict, Tuple

# --- Import existing, well-defined components from the project ---
from augmentations import SegmentationAugmentation
from optimizer import create_optimizer_and_scheduler
from losses import CoralLoss
from train_baseline import (
    get_class_names,
    CoralscapesSingleTaskDataset,
    SingleTaskMetrics,
    ConfigBaseline,
)
import reporting

# Tell Optuna to shut up about experimental features unless it's an error
optuna.logging.set_verbosity(optuna.logging.WARNING)


# === 1. MODIFIED CONFIGURATION FOR ADVANCED TUNING ===
class TuneConfigAdvanced(ConfigBaseline):
    """Mutable configuration for the advanced Optuna trials."""
    # Increased batch size, removed gradient accumulation
    BATCH_SIZE = 16
    # GRADIENT_ACCUMULATION_STEPS removed

    # Placeholders for tunable augmentation parameters
    CROP_SCALE_MIN: float = 0.5
    ROTATION_DEGREES: int = 15
    JITTER_PARAMS: Dict[str, float] = None

    # Placeholders for tunable loss parameters
    PRIMARY_LOSS_TYPE: str = 'focal'
    DICE_SMOOTH: float = 1.0

    # Shorter epoch count for faster tuning trials
    NUM_EPOCHS_TUNE: int = 15
    # Full epoch count for the final run
    NUM_EPOCHS_FINAL: int = 50


# === 2. REFACTORED TRAINING & VALIDATION ENGINE ===

def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn, scaler, device):
    """Modular training function for a single epoch without gradient accumulation."""
    model.train()
    total_loss = 0
    loop = tqdm(enumerate(dataloader), desc="Training Trial", total=len(dataloader), leave=False)

    for i, batch in loop:
        images, masks = batch['image'].to(device, non_blocking=True), batch['mask'].to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=str(device), dtype=torch.float16):
            outputs = model(pixel_values=images)
            upsampled_logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = loss_fn(upsampled_logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        current_loss = loss.item()
        total_loss += current_loss
        loop.set_postfix(loss=current_loss)
    return total_loss / len(dataloader)


def validate_one_epoch(model, dataloader, loss_fn, metrics_calculator, device):
    """Modular validation function for a single epoch (unchanged)."""
    model.eval()
    total_val_loss = 0.0
    metrics_calculator.reset()
    loop = tqdm(dataloader, desc="Validating Trial", leave=False)
    with torch.no_grad():
        for batch in loop:
            images, masks = batch['image'].to(device, non_blocking=True), batch['mask'].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=str(device), dtype=torch.float16):
                outputs = model(pixel_values=images)
                upsampled_logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                total_val_loss += loss_fn(upsampled_logits, masks).item()
                metrics_calculator.update(upsampled_logits, masks)

    avg_loss = total_val_loss / len(dataloader)
    metrics = metrics_calculator.compute()
    return avg_loss, metrics['mIoU']


def run_training_session(config, trial: optuna.Trial = None, is_final_run: bool = False):
    """A complete training and validation session, adapted for the new config."""
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Loading ---
    hf_dataset = load_dataset("EPFL-ECEO/coralscapes", trust_remote_code=True)
    if config.DEBUG:
        train_subset = hf_dataset['train'].select(range(config.BATCH_SIZE * 4))
        val_subset = hf_dataset['validation'].select(range(config.BATCH_SIZE * 4))
        hf_dataset = DatasetDict({'train': train_subset, 'validation': val_subset})

    train_augs = SegmentationAugmentation(
        patch_size=config.PATCH_SIZE,
        crop_scale=(config.CROP_SCALE_MIN, 1.0),
        rotation_degrees=config.ROTATION_DEGREES,
        jitter_params=config.JITTER_PARAMS
    )
    train_dataset = CoralscapesSingleTaskDataset(hf_dataset, 'train', train_augs, config.PATCH_SIZE)
    val_dataset = CoralscapesSingleTaskDataset(hf_dataset, 'validation', None, config.PATCH_SIZE)
    train_loader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    # --- Model, Loss, Optimizer ---
    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(
        config.ENCODER_NAME, id2label={i: i for i in range(config.NUM_CLASSES)},
        semantic_loss_ignore_index=0, ignore_mismatched_sizes=True
    ).to(config.DEVICE)

    loss_fn = CoralLoss(
        primary_loss_type=config.PRIMARY_LOSS_TYPE,
        hybrid_alpha=config.HYBRID_ALPHA,
        focal_gamma=config.FOCAL_GAMMA,
        dice_smooth=config.DICE_SMOOTH
    ).to(config.DEVICE)

    num_epochs = config.NUM_EPOCHS_FINAL if is_final_run else config.NUM_EPOCHS_TUNE
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * config.WARMUP_STEPS_RATIO)

    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config.LEARNING_RATE, config.WEIGHT_DECAY, config.ADAM_BETAS, total_steps, warmup_steps
    )
    scaler = torch.amp.GradScaler()
    metrics_calculator = SingleTaskMetrics(config.NUM_CLASSES, config.DEVICE)

    # --- Training Loop ---
    best_miou = -1.0
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, scaler, config.DEVICE)
        avg_val_loss, val_miou = validate_one_epoch(model, val_loader, loss_fn, metrics_calculator, config.DEVICE)

        if is_final_run:
            print(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {val_miou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            if is_final_run:
                torch.save(model.state_dict(), os.path.join(output_dir, config.BEST_MODEL_NAME))
                print(f"  >>> New best model saved with mIoU: {best_miou:.4f}")

        if trial:
            trial.report(val_miou, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # --- Final Reporting for the best model ---
    if is_final_run:
        print("\n--- Generating Final Reports for the Best Model ---")
        class_names = get_class_names()
        best_model_path = os.path.join(output_dir, config.BEST_MODEL_NAME)
        model.load_state_dict(torch.load(best_model_path))

        _, _ = validate_one_epoch(model, val_loader, loss_fn, metrics_calculator, config.DEVICE)
        final_metrics = metrics_calculator.compute()

        class_iou_dict = {name: score for name, score in zip(class_names[1:], final_metrics['per_class_iou'][1:])}
        reporting.plot_per_class_iou_bar_chart(class_iou_dict, "Tuned Baseline", save_path=f"{output_dir}/tuned_per_class_iou.png")

        cm_for_plot = metrics_calculator.cm.cpu().numpy()[1:, 1:]
        reporting.plot_confusion_matrix(cm_for_plot, class_names[1:], "Tuned Baseline", save_path=f"{output_dir}/tuned_confusion_matrix.png")

        # 4. Qualitative Results Grid
        print("Generating qualitative results grid...")
        model.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            images = batch['image'].to(config.DEVICE)
            gt_masks = batch['mask']
            with torch.amp.autocast(device_type=str(config.DEVICE), dtype=torch.float16):
                outputs = model(pixel_values=images)
                # --- FIX: Upsample logits for qualitative visualization ---
                pred_logits = F.interpolate(
                    outputs.logits,
                    size=gt_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                # --------------------------------------------------------
            pred_masks = torch.argmax(pred_logits, dim=1)

            reporting.plot_qualitative_grid(
                images.cpu(), 
                {'Ground Truth': gt_masks}, 
                {'Prediction': pred_masks.cpu()},
                save_path=f"{output_dir}/tuned_qualitative_grid.png"
            )

    return best_miou


# === 3. OPTUNA OBJECTIVE FUNCTION ===
def objective(trial: optuna.Trial) -> float:
    """The function for Optuna to optimize with an expanded search space."""
    config = TuneConfigAdvanced()
    config.DEBUG = True

    # --- Define Hyperparameter Search Space ---
    # Optimizer and Scheduler
    config.LEARNING_RATE = trial.suggest_float("learning_rate", 1e-5, 2e-4, log=True)
    config.WEIGHT_DECAY = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    config.WARMUP_STEPS_RATIO = trial.suggest_float("warmup_ratio", 0.05, 0.25)

    # Loss parameters (now includes loss type and dice smooth)
    config.PRIMARY_LOSS_TYPE = trial.suggest_categorical("primary_loss_type", ["focal", "cross_entropy"])
    config.HYBRID_ALPHA = trial.suggest_float("hybrid_alpha", 0.3, 0.9)
    config.DICE_SMOOTH = trial.suggest_float("dice_smooth", 0.5, 1.5)

    # Conditional hyperparameter: focal_gamma is only relevant for focal loss
    if config.PRIMARY_LOSS_TYPE == 'focal':
        config.FOCAL_GAMMA = trial.suggest_float("focal_gamma", 1.0, 3.5)
    else:
        config.FOCAL_GAMMA = 2.0  # Default value, not used by CrossEntropyLoss

    # Stronger Augmentation parameters
    config.ROTATION_DEGREES = trial.suggest_int("rotation_degrees", 10, 90)
    config.CROP_SCALE_MIN = trial.suggest_float("crop_scale_min", 0.2, 0.8)
    config.JITTER_PARAMS = {
        'brightness': trial.suggest_float("jitter_brightness", 0.1, 0.5),
        'contrast': trial.suggest_float("jitter_contrast", 0.1, 0.5),
        'saturation': trial.suggest_float("jitter_saturation", 0.1, 0.5),
        'hue': trial.suggest_float("jitter_hue", 0.05, 0.2)
    }

    best_miou = run_training_session(config, trial=trial)
    return best_miou


# === 4. MAIN EXECUTION SCRIPT ===
if __name__ == "__main__":

    # --- PART 1: HYPERPARAMETER TUNING ---
    print("--- Starting Advanced Optuna Hyperparameter Study ---")
    study_name = "baseline-tuning"
    storage_name = f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=5)
    )

    study.optimize(objective, n_trials=75, timeout=10800) # Run for 75 trials or 3 hours

    # --- PART 2: REPORTING THE STUDY RESULTS ---
    print("\n\n--- Optuna Study Report ---")
    print(f"Study '{study_name}' complete.")
    print(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    print(f"Best trial value (mIoU): {best_trial.value:.4f}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  - {key}: {value}")

    report_dir = "optuna_reports_advanced"
    os.makedirs(report_dir, exist_ok=True)
    try:
        fig_history = optuna.visualization.plot_optimization_history(study)
        fig_history.write_image(os.path.join(report_dir, "optimization_history.png"))

        fig_importance = optuna.visualization.plot_param_importances(study)
        fig_importance.write_image(os.path.join(report_dir, "param_importances.png"))
        print(f"\nOptuna report plots saved to '{report_dir}' directory.")
    except (ValueError, ImportError) as e:
        print(f"Could not generate plots: {e}. Skipping plot generation.")


    # --- PART 3: RUNNING FINAL TRAINING WITH BEST PARAMETERS ---
    print("\n\n--- Starting Final Training with Best Hyperparameters ---")

    final_config = TuneConfigAdvanced()
    final_config.DEBUG = False
    final_config.OUTPUT_DIR = "training_outputs/baseline_tuned_advanced"

    # Update the config with the best found parameters from the trial
    final_config.LEARNING_RATE = best_trial.params["learning_rate"]
    final_config.WEIGHT_DECAY = best_trial.params["weight_decay"]
    final_config.WARMUP_STEPS_RATIO = best_trial.params["warmup_ratio"]
    final_config.PRIMARY_LOSS_TYPE = best_trial.params["primary_loss_type"]
    final_config.HYBRID_ALPHA = best_trial.params["hybrid_alpha"]
    final_config.DICE_SMOOTH = best_trial.params["dice_smooth"]
    if "focal_gamma" in best_trial.params:
        final_config.FOCAL_GAMMA = best_trial.params["focal_gamma"]
    final_config.ROTATION_DEGREES = best_trial.params["rotation_degrees"]
    final_config.CROP_SCALE_MIN = best_trial.params["crop_scale_min"]
    final_config.JITTER_PARAMS = {
        'brightness': best_trial.params["jitter_brightness"],
        'contrast': best_trial.params["jitter_contrast"],
        'saturation': best_trial.params["jitter_saturation"],
        'hue': best_trial.params["jitter_hue"]
    }

    run_training_session(final_config, is_final_run=True)

    print("\n--- Advanced Tuned Baseline Training Complete! ---")
    print(f"Final model and reports are saved in '{final_config.OUTPUT_DIR}'")