import torch
from tqdm import tqdm
import os
from collections import defaultdict
import optuna
from pathlib import Path
import json
from typing import Dict, Any

from .inference import SlidingWindowInferrer
from .metrics import AbstractCoralMetrics
from coral_mtl.utils.metrics_storer import MetricsStorer

class Trainer:
    """
    A generic training and validation engine for Coral-MTL and baseline models.

    This class encapsulates the entire training loop, including epoch management,
    forward/backward passes, optimization, metric calculation, checkpointing,
    and optional integration with Optuna for hyperparameter tuning.
    """
    def __init__(self, model, train_loader, val_loader, loss_fn, metrics_calculator: AbstractCoralMetrics,
                 metrics_storer: MetricsStorer, optimizer, scheduler, config, trial: optuna.Trial = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metrics_calculator = metrics_calculator
        self.metrics_storer = metrics_storer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.trial = trial # For Optuna integration
        
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        
        # Configure mixed precision based on config and device compatibility
        self.use_mixed_precision = getattr(config, 'use_mixed_precision', False) and (self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler(enabled=self.use_mixed_precision)
        
        print(f"Training device: {self.device}")
        print(f"Mixed precision (FP16) enabled: {self.use_mixed_precision}")
        
        self.best_metric = -1.0
        self.training_log = defaultdict(list)
        self.validation_log = defaultdict(list)

    def _train_one_epoch(self):
        """Executes a single training epoch."""
        self.model.train()
        loop = tqdm(self.train_loader, desc=f"Training Epoch", leave=False)
        
        for i, batch in enumerate(loop):
            # The 'masks' key holds the appropriately transformed GT for the loss function
            images = batch['image'].to(self.device, non_blocking=True)
            # Use 'masks' for MTL (dict) or 'mask' for baseline (tensor)
            masks_for_loss = batch.get('masks', batch.get('mask'))
            
            if isinstance(masks_for_loss, dict):
                masks_for_loss = {k: v.to(self.device, non_blocking=True) for k, v in masks_for_loss.items()}
            else:
                masks_for_loss = masks_for_loss.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_mixed_precision):
                predictions = self.model(images)
                loss_dict_or_tensor = self.loss_fn(predictions, masks_for_loss)
                
                # Standardize loss handling for logging
                if isinstance(loss_dict_or_tensor, dict):
                    total_loss = loss_dict_or_tensor['total_loss']
                    for key, val in loss_dict_or_tensor.items():
                        if torch.is_tensor(val): self.training_log[key].append(val.item())
                else:
                    total_loss = loss_dict_or_tensor
                    self.training_log['total_loss'].append(total_loss.item())
                
                # Handle gradient accumulation
                total_loss = total_loss / self.config.gradient_accumulation_steps

            self.scaler.scale(total_loss).backward()

            if (i + 1) % self.config.gradient_accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            self.training_log['lr'].append(self.scheduler.get_last_lr()[0])
            loop.set_postfix(loss=(total_loss.item() * self.config.gradient_accumulation_steps))

    def _validate_one_epoch(self, epoch: int = None) -> Dict[str, Any]:
        """
        Executes a single validation epoch using sliding window inference.
        It computes a full metrics report and stores per-image confusion matrices.
        """
        self.model.eval()
        self.metrics_calculator.reset()
        
        inferrer = SlidingWindowInferrer(
            model=self.model,
            patch_size=self.config.patch_size,
            stride=self.config.inference_stride,
            device=self.device,
            batch_size=self.config.inference_batch_size
        )
        
        loop = tqdm(self.val_loader, desc=f"Validation", leave=False)
        with torch.no_grad():
            for batch in loop:
                # The validation loader yields batches of full images.
                # The `original_mask` and `image_id` are now essential.
                batch_images = batch['image']
                original_masks = batch['original_mask'].to(self.device)
                image_ids = batch['image_id']

                # Perform inference on the batch of full-size images
                # Note: We need to loop through each image in the batch since predict takes single images
                batch_predictions = {}
                for idx, single_image in enumerate(batch_images):
                    single_predictions = inferrer.predict(single_image)
                    if idx == 0:
                        # Initialize batch_predictions with the right structure
                        for task_name in single_predictions:
                            batch_predictions[task_name] = []
                    for task_name, logits in single_predictions.items():
                        batch_predictions[task_name].append(logits)
                
                # Stack predictions for batch processing
                stitched_predictions_logits = {
                    task_name: torch.stack(task_logits, dim=0)
                    for task_name, task_logits in batch_predictions.items()
                }

                # Update metrics with the required data payload
                self.metrics_calculator.update(
                    predictions=stitched_predictions_logits,
                    original_targets=original_masks,
                    image_ids=image_ids
                )
        
        # After the loop, store all buffered per-image CMs from this epoch
        for img_id, cms, predictions in self.metrics_calculator.per_image_cms_buffer:
            self.metrics_storer.store_per_image_cms(img_id, cms, predictions, is_testing=False, epoch=epoch)
        
        # Clear the buffer in preparation for the next validation epoch
        self.metrics_calculator.per_image_cms_buffer.clear()
        
        # Finally, compute the aggregate metrics for the entire validation set
        return self.metrics_calculator.compute()

    def _get_metric_from_report(self, report: Dict[str, Any], key_path: str) -> float:
        """Accesses a nested key in the report, e.g., 'tasks.genus.grouped.mIoU'"""
        try:
            keys = key_path.split('.')
            value = report
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            print(f"Warning: Metric '{key_path}' not found in report. Returning -1.0")
            return -1.0

    def train(self):
        """The main training loop orchestrating epochs, validation, and checkpointing."""
        print(f"--- Starting Training for {self.config.epochs} epochs ---")
        
        try:
            # Open the JSONL file handle once for the entire training run
            self.metrics_storer.open_for_run(is_testing=False)
            
            for epoch in range(self.config.epochs):
                print(f"\n===== Epoch {epoch+1}/{self.config.epochs} =====")
                self._train_one_epoch()
                val_metrics_report = self._validate_one_epoch(epoch + 1)
                
                # Store the flattened summary metrics for this epoch
                self.metrics_storer.store_epoch_history(val_metrics_report, epoch + 1)
                
                # Use the specified metric from the config for model selection
                current_metric = self._get_metric_from_report(val_metrics_report, self.config.model_selection_metric)
                
                print(f"Epoch {epoch+1} Summary:")
                print(f"  Validation Metric ({self.config.model_selection_metric}): {current_metric:.4f} (Best: {max(self.best_metric, current_metric):.4f})")

                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    os.makedirs(self.config.output_dir, exist_ok=True)
                    save_path = os.path.join(self.config.output_dir, "best_model.pth")
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  >>> New best model saved to {save_path}")

                if self.trial:
                    self.trial.report(current_metric, epoch)
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
        
        finally:
            # Ensure the file handle is always closed safely
            self.metrics_storer.close()
        
        print("\n--- Training Complete ---")
        # The history is now managed by the storer, so we don't return it directly
        return self.best_metric