import torch
from tqdm import tqdm
import os
from collections import defaultdict
import optuna

from .inference import SlidingWindowInferrer

class Trainer:
    """
    A generic training and validation engine for Coral-MTL and baseline models.

    This class encapsulates the entire training loop, including epoch management,
    forward/backward passes, optimization, metric calculation, checkpointing,
    and optional integration with Optuna for hyperparameter tuning.
    """
    def __init__(self, model, train_loader, val_loader, loss_fn, metrics_calculator,
                 optimizer, scheduler, config, trial: optuna.Trial = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metrics_calculator = metrics_calculator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.trial = trial # For Optuna integration
        
        self.device = config.DEVICE
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        
        self.best_metric = -1.0
        self.training_log = defaultdict(list)
        self.validation_log = defaultdict(list)

    def _train_one_epoch(self):
        """Executes a single training epoch."""
        self.model.train()
        loop = tqdm(enumerate(self.train_loader), desc=f"Training Epoch", total=len(self.train_loader), leave=False)
        self.optimizer.zero_grad()
        
        for i, batch in loop:
            # The structure of 'masks' determines if it's MTL or single-task
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['masks'] if isinstance(batch.get('masks'), dict) else batch['mask']
            
            if isinstance(masks, dict): # MTL case
                masks = {k: v.to(self.device, non_blocking=True) for k, v in masks.items()}
            else: # Baseline case
                masks = masks.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
                predictions = self.model(images)
                loss_dict_or_tensor = self.loss_fn(predictions, masks)
                
                # Unify loss handling
                if isinstance(loss_dict_or_tensor, dict):
                    total_loss = loss_dict_or_tensor['total_loss']
                    for key, val in loss_dict_or_tensor.items():
                        if torch.is_tensor(val):
                            self.training_log[key].append(val.item())
                else:
                    total_loss = loss_dict_or_tensor
                    self.training_log['total_loss'].append(total_loss.item())

                total_loss = total_loss / self.config.GRADIENT_ACCUMULATION_STEPS

            self.scaler.scale(total_loss).backward()

            if (i + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(self.train_loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            self.training_log['lr'].append(self.scheduler.get_last_lr()[0])
            loop.set_postfix(loss=(total_loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS))

    def _validate_one_epoch(self):
        """
        Executes a single validation epoch using sliding window inference.
        This method implements the procedure from Spec Section 8.2, ensuring
        that validation metrics are a reliable proxy for test performance.
        """
        self.model.eval()
        self.metrics_calculator.reset()
        
        # Instantiate the inferrer for validation as per Spec Section 8.2
        # This simulates the final testing pipeline for reliable model selection.
        inferrer = SlidingWindowInferrer(
            model=self.model,
            patch_size=self.config.PATCH_SIZE,
            stride=self.config.INFERENCE_STRIDE,
            device=self.device,
            batch_size=self.config.INFERENCE_BATCH_SIZE
        )
        
        loop = tqdm(self.val_loader, desc=f"Validation", leave=False)
        with torch.no_grad():
            for batch in loop:
                # The validation loader should yield full images, typically with batch size 1.
                # We loop through the batch in case batch_size > 1.
                batch_images = batch['image']
                
                for i in range(batch_images.shape[0]):
                    full_image = batch_images[i] # Shape: (C, H, W)
                    
                    # Perform sliding window inference to get stitched prediction logits
                    # Implements Spec Section 8.2, step 3
                    stitched_predictions = inferrer.predict(full_image)

                    # Prepare ground truth masks for metric calculation
                    if 'masks' in batch: # MTL case
                        gt_masks = {k: v[i].to(self.device) for k, v in batch['masks'].items()}
                        # Unsqueeze predictions and masks for metric calculator (expects batch dim)
                        stitched_predictions = {k: v.unsqueeze(0) for k, v in stitched_predictions.items()}
                        gt_masks_batched = {k: v.unsqueeze(0) for k, v in gt_masks.items()}
                    else: # Baseline case
                        gt_masks = batch['mask'][i].to(self.device)
                        # Unsqueeze predictions and masks for metric calculator (expects batch dim)
                        stitched_predictions = {k: v.unsqueeze(0) for k, v in stitched_predictions.items()}
                        gt_masks_batched = gt_masks.unsqueeze(0)
                
                    # Update metrics with full-resolution stitched maps
                    # Implements Spec Section 8.2, step 4
                    self.metrics_calculator.update(stitched_predictions, gt_masks_batched)
        
        # Note: Loss is not calculated during validation, as the focus is on metrics
        # from the realistic sliding window inference pipeline.
        val_metrics = self.metrics_calculator.compute()
        
        for key, value in val_metrics.items():
            self.validation_log[key].append(value)

        return val_metrics

    def train(self):
        """The main training loop orchestrating epochs, validation, and checkpointing."""
        print(f"--- Starting Training for {self.config.NUM_EPOCHS} epochs ---")
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n===== Epoch {epoch+1}/{self.config.NUM_EPOCHS} =====")
            self._train_one_epoch()
            val_metrics = self._validate_one_epoch()
            
            # Use 'H-Mean' for MTL, 'mIoU' for baseline
            # This is the key metric for model selection as per Spec Section 7.3.1
            current_metric = val_metrics.get('H-Mean', val_metrics.get('mIoU', -1.0))
            
            print(f"Epoch {epoch+1} Summary:")
            # We no longer report validation loss, focusing on the more meaningful metrics.
            print(f"  Validation Metric ({self.config.MODEL_SELECTION_METRIC}): {current_metric:.4f} (Best: {max(self.best_metric, current_metric):.4f})")

            if current_metric > self.best_metric:
                self.best_metric = current_metric
                os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
                save_path = os.path.join(self.config.OUTPUT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"  >>> New best model saved to {save_path}")

            if self.trial:
                self.trial.report(current_metric, epoch)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        
        print("\n--- Training Complete ---")
        return self.best_metric, self.training_log, self.validation_log