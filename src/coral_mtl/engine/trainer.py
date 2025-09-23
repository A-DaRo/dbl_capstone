import torch
from tqdm import tqdm
import os
from collections import defaultdict
import optuna
from pathlib import Path
import json
from typing import Dict, Any

from .inference import SlidingWindowInferrer
from ..metrics.metrics import AbstractCoralMetrics
from ..metrics.metrics_storer import MetricsStorer, AdvancedMetricsProcessor

class Trainer:
    """
    A generic training and validation engine for Coral-MTL and baseline models.

    This class encapsulates the entire training loop, including epoch management,
    forward/backward passes, optimization, metric calculation, checkpointing,
    and optional integration with Optuna for hyperparameter tuning.
    """
    def __init__(self, model, train_loader, val_loader, loss_fn, metrics_calculator: AbstractCoralMetrics,
                 metrics_storer: MetricsStorer, optimizer, scheduler, config, trial: optuna.Trial = None,
                 metrics_processor: AdvancedMetricsProcessor = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metrics_calculator = metrics_calculator
        self.metrics_storer = metrics_storer
        self.metrics_processor = metrics_processor  # Tier 2/3 processor
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
        if self.metrics_processor:
            print(f"Advanced metrics processing enabled with {self.metrics_processor.num_cpu_workers} workers")
        
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
        It computes a full metrics report and dispatches data to both Tier 1 and Tier 2 systems.
        """
        self.model.eval()
        
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
                batch_images = batch['image'].to(self.device, non_blocking=True)
                original_masks = batch['original_mask'].to(self.device, non_blocking=True)
                image_ids = batch['image_id']

                # Perform inference on the entire batch of full-size images at once
                stitched_predictions_logits = inferrer.predict(batch_images)

                # Handle different model types for metrics calculator
                # MTL models expect dictionary, baseline models expect single tensor
                if len(stitched_predictions_logits) == 1 and 'segmentation' in stitched_predictions_logits:
                    # This is a baseline model - extract the single tensor for CoralMetrics
                    predictions_for_metrics = stitched_predictions_logits['segmentation']
                    predictions_logits_for_tier1 = stitched_predictions_logits['segmentation']
                else:
                    # This is an MTL model - pass the full dictionary for CoralMTLMetrics
                    predictions_for_metrics = stitched_predictions_logits
                    predictions_logits_for_tier1 = stitched_predictions_logits

                # Tier 1: Update metrics calculator with logits for GPU-based computation
                self.metrics_calculator.update(
                    predictions=predictions_for_metrics,
                    original_targets=original_masks,
                    image_ids=image_ids,
                    epoch=epoch,
                    predictions_logits=predictions_logits_for_tier1,
                    store_per_image=False  # Skip expensive disk I/O during training
                )
                
                # Tier 2: Dispatch jobs to CPU worker pool (if enabled)
                if self.metrics_processor:
                    batch_size = original_masks.shape[0]
                    
                    # Get prediction masks from logits for Tier 2 processing
                    if isinstance(predictions_for_metrics, dict):
                        # MTL model - use first task or create global prediction
                        first_task_logits = next(iter(predictions_for_metrics.values()))
                        pred_masks = torch.argmax(first_task_logits, dim=1)
                    else:
                        # Baseline model
                        pred_masks = torch.argmax(predictions_for_metrics, dim=1)
                    
                    # Dispatch each image in the batch
                    for i in range(batch_size):
                        self.metrics_processor.dispatch_image_job(
                            image_id=image_ids[i],
                            pred_mask_tensor=pred_masks[i],
                            target_mask_tensor=original_masks[i]
                        )
        
        # After the loop, compute the aggregate metrics for the entire validation set (Tier 1 only)
        return self.metrics_calculator.compute()

    def _should_validate(self, epoch):
        """Determine if validation should run this epoch to save time."""
        # Validate every epoch for first 3 epochs
        if epoch <= 3:
            return True
        # Then every 2 epochs until epoch 10
        elif epoch <= 10:
            return epoch % 2 == 0
        # Finally every 3 epochs
        else:
            return epoch % 3 == 0

    def _get_metric_from_report(self, report: Dict[str, Any], key_path: str) -> float:
        """Accesses a nested key in the report, e.g., 'tasks.genus.grouped.mIoU' or 'global.BIoU'"""
        try:
            # First check if the metric is in optimization_metrics (the flat dictionary)
            if 'optimization_metrics' in report and key_path in report['optimization_metrics']:
                return report['optimization_metrics'][key_path]
            
            # If not found in optimization_metrics, try the old nested approach
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
            
            # Start the advanced metrics processor if available
            if self.metrics_processor:
                self.metrics_processor.start()
            
            for epoch in range(self.config.epochs):
                print(f"\n===== Epoch {epoch+1}/{self.config.epochs} =====")
                self._train_one_epoch()
                
                # Only validate if scheduled (saves significant time)
                if self._should_validate(epoch + 1):
                    # Reset metrics calculator at the start of each validation epoch
                    self.metrics_calculator.reset()
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
                else:
                    print(f"Epoch {epoch+1} Summary: Validation skipped (scheduled for every few epochs)")
                    # Still update trial for Optuna if needed
                    if self.trial:
                        self.trial.report(self.best_metric, epoch)  # Report current best
        
        finally:
            # Ensure all resources are always closed safely
            self.metrics_storer.close()
            if self.metrics_processor:
                self.metrics_processor.shutdown()
        
        print("\n--- Training Complete ---")
        # The history is now managed by the storer, so we don't return it directly
        return self.best_metric