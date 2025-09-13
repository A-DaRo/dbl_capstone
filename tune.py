import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
import math
from transformers import SegformerForSemanticSegmentation

# --- Import refactored components ---
from coral_mtl.data.augmentations import SegmentationAugmentation
from coral_mtl.engine.optimizer import create_optimizer_and_scheduler
from coral_mtl.engine.losses import CoralLoss
from coral_mtl.engine.trainer import Trainer
# Note: For a real project, baseline-specific data/metrics would also be in src/
from train_baseline import CoralscapesSingleTaskDataset, SingleTaskMetrics, ConfigBaseline

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Use the config as a mutable object for trial parameters
class TuneConfig(ConfigBaseline):
    pass

def objective(trial: optuna.Trial) -> float:
    """The function for Optuna to optimize."""
    config = TuneConfig()
    config.DEBUG = True 
    config.NUM_EPOCHS = 10 # Shorten epochs for tuning

    # --- Define Hyperparameter Search Space ---
    config.LEARNING_RATE = trial.suggest_float("learning_rate", 1e-5, 2e-4, log=True)
    config.WEIGHT_DECAY = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    config.HYBRID_ALPHA = trial.suggest_float("hybrid_alpha", 0.3, 0.9)
    # ... add other parameters to tune ...

    # --- Setup components for a baseline run ---
    hf_dataset = load_dataset("EPFL-ECEO/coralscapes", split="train[:1%]") # Use tiny subset
    
    train_augs = SegmentationAugmentation(patch_size=config.PATCH_SIZE)
    train_dataset = CoralscapesSingleTaskDataset(hf_dataset, 'train', train_augs, config.PATCH_SIZE)
    val_dataset = CoralscapesSingleTaskDataset(hf_dataset, 'validation', None, config.PATCH_SIZE)
    train_loader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, config.BATCH_SIZE, shuffle=False)

    model = SegformerForSemanticSegmentation.from_pretrained(
        config.ENCODER_NAME, num_labels=config.NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(config.DEVICE)
    
    # This wrapper is needed because the baseline model outputs an object, not a tensor
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, images):
            outputs = self.model(pixel_values=images).logits
            return F.interpolate(outputs, size=images.shape[-2:], mode='bilinear', align_corners=False)
    
    wrapped_model = ModelWrapper(model)

    loss_fn = CoralLoss(primary_loss_type='focal', hybrid_alpha=config.HYBRID_ALPHA).to(config.DEVICE)
    
    total_steps = len(train_loader) * config.NUM_EPOCHS
    optimizer, scheduler = create_optimizer_and_scheduler(model, config.LEARNING_RATE, config.WEIGHT_DECAY,
                                                          (0.9, 0.999), total_steps, int(total_steps * 0.1))
    
    metrics_calculator = SingleTaskMetrics(config.NUM_CLASSES, config.DEVICE)

    # --- Instantiate and Run the Trainer ---
    trainer = Trainer(wrapped_model, train_loader, val_loader, loss_fn, metrics_calculator,
                      optimizer, scheduler, config, trial=trial)
    
    best_miou, _, _ = trainer.train()
    return best_miou

if __name__ == "__main__":
    study_name = "baseline-tuning"
    storage_name = f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )

    study.optimize(objective, n_trials=20)

    print("\n--- Optuna Study Complete ---")
    best_trial = study.best_trial
    print(f"Best trial mIoU: {best_trial.value:.4f}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  - {key}: {value}")

    # The logic for running a final, full training session with the best
    # parameters would follow here, using the same setup as the 'objective'
    # function but with `is_final_run=True` and loading the best params.