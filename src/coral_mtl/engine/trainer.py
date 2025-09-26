import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from collections import defaultdict
import optuna
from pathlib import Path
import json
from typing import Dict, Any, List, Optional, Tuple

from .inference import SlidingWindowInferrer
from ..metrics.metrics import AbstractCoralMetrics
from ..metrics.metrics_storer import MetricsStorer, AdvancedMetricsProcessor
from .gradient_strategies import GradientUpdateStrategy

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
        pcgrad_cfg = getattr(config, 'pcgrad', None)
        self.pcgrad_enabled = bool(getattr(pcgrad_cfg, 'enabled', False)) if pcgrad_cfg else False
        if self.pcgrad_enabled:
            if getattr(config, 'gradient_accumulation_steps', 1) != 1:
                raise ValueError("PCGrad currently requires gradient_accumulation_steps == 1")
            self.use_mixed_precision = False
        self.scaler = torch.amp.GradScaler(enabled=self.use_mixed_precision)
        self.weighting_strategy = getattr(self.loss_fn, 'weighting_strategy', None)
        self._shared_params_registered = False
        if self.pcgrad_enabled:
            from .pcgrad import PCGrad
            self.pcgrad = PCGrad(self.optimizer)
        else:
            self.pcgrad = None
        
        print(f"Training device: {self.device}")
        print(f"Mixed precision (FP16) enabled: {self.use_mixed_precision}")
        if self.metrics_processor:
            print(f"Advanced metrics processing enabled with {self.metrics_processor.num_cpu_workers} workers")
        
        self.best_metric = -1.0
        self.training_log = defaultdict(list)
        self.validation_log = defaultdict(list)
        self.global_step = 0
        self.current_epoch = 0
        self._log_frequency = getattr(self.config, 'log_frequency', 50)

    def _maybe_register_gradnorm_params(self):
        if self.weighting_strategy and getattr(self.weighting_strategy, 'requires_manual_backward_update', None):
            if self.weighting_strategy.requires_manual_backward_update() and not self._shared_params_registered:
                if hasattr(self.weighting_strategy, 'register_shared_parameters'):
                    params = [p for p in self.model.parameters() if p.requires_grad]
                    self.weighting_strategy.register_shared_parameters(params)
                self._shared_params_registered = True

    def _train_one_epoch(self, epoch_index: int):
        """Executes a single training epoch and returns averaged loss components (namespaced)."""
        self.model.train()
        self._maybe_register_gradnorm_params()
        loop = tqdm(self.train_loader, desc=f"Training Epoch", leave=False)
        epoch_train_losses = defaultdict(list)

        for i, batch in enumerate(loop):
            diagnostics_payload: Dict[str, Any] = {}
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
                # Branch depending on strategy type
                if isinstance(getattr(self.loss_fn, 'weighting_strategy', None), GradientUpdateStrategy):
                    # Gradient-based strategy path
                    unweighted = self.loss_fn.compute_unweighted_losses(predictions, masks_for_loss if isinstance(masks_for_loss, dict) else {k: masks_for_loss for k in getattr(self.loss_fn, 'primary_tasks', [])})
                    diagnostics_payload['unweighted_losses'] = self._tensor_dict_to_floats(unweighted)
                    per_task_grads = {}
                    params = [p for p in self.model.parameters() if p.requires_grad]
                    flat_shapes = [p.shape for p in params]
                    numels = [p.numel() for p in params]
                    for t_name, t_loss in unweighted.items():
                        self.optimizer.zero_grad()
                        (t_loss / self.config.gradient_accumulation_steps).backward(retain_graph=True)
                        grads_flat = []
                        for p in params:
                            if p.grad is None:
                                grads_flat.append(torch.zeros(p.numel(), device=self.device, dtype=p.dtype))
                            else:
                                grads_flat.append(p.grad.detach().clone().flatten())
                        per_task_grads[t_name] = torch.cat(grads_flat)
                    # combine
                    update_vec = self.loss_fn.weighting_strategy.compute_update_vector(per_task_grads)
                    # Diagnostics captured after strategy computation
                    grad_norm, grad_cos = self._compute_gradient_diagnostics(per_task_grads)
                    diagnostics_payload['gradient_norm'] = grad_norm
                    if grad_cos:
                        diagnostics_payload['gradient_cosine_similarity'] = grad_cos
                    strategy_diagnostics = getattr(self.loss_fn.weighting_strategy, 'get_last_diagnostics', None)
                    if callable(strategy_diagnostics):
                        diagnostics_payload.update(self.loss_fn.weighting_strategy.get_last_diagnostics())
                    diagnostics_payload.setdefault('gradient_update_norm', float(update_vec.norm().detach().item()))
                    # apply combined gradient
                    self.optimizer.zero_grad()
                    offset = 0
                    for p, n in zip(params, numels):
                        segment = update_vec[offset: offset + n].view(p.shape)
                        p.grad = segment.clone()
                        offset += n
                    loss_dict_or_tensor = {'total_loss': update_vec.norm()}  # placeholder scalar for logging
                else:
                    loss_dict_or_tensor = self.loss_fn(predictions, masks_for_loss)
                    if isinstance(loss_dict_or_tensor, dict):
                        diagnostics_payload.update(self._extract_weighting_diagnostics())

            # Standardize loss handling for logging (outside autocast)
            if isinstance(loss_dict_or_tensor, dict):
                total_loss = loss_dict_or_tensor['total_loss']
                for key, val in loss_dict_or_tensor.items():
                    if torch.is_tensor(val):
                        scalar = float(val.detach().item())
                        self.training_log[key].append(scalar)
                        epoch_train_losses[key].append(scalar)
            else:
                total_loss = loss_dict_or_tensor
                scalar = float(total_loss.detach().item())
                self.training_log['total_loss'].append(scalar)
                epoch_train_losses['total_loss'].append(scalar)

            if self.weighting_strategy and self.weighting_strategy.requires_manual_backward_update():
                self.weighting_strategy.manual_backward_update(self.model)

            raw_total_loss_value = float(total_loss.detach().item())

            if self.pcgrad_enabled:
                self.optimizer.zero_grad()
                self._apply_pcgrad(loss_dict_or_tensor)
                self.optimizer.zero_grad()
                self.scheduler.step()
            else:
                total_loss = total_loss / self.config.gradient_accumulation_steps
                self.scaler.scale(total_loss).backward()

                if (i + 1) % self.config.gradient_accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

            self.training_log['lr'].append(self.scheduler.get_last_lr()[0])
            loop.set_postfix(loss=raw_total_loss_value)

            self._maybe_record_loss_diagnostics(epoch_index, diagnostics_payload)
            self.global_step += 1

        # Aggregate means and namespace
        raw_means = {k: float(sum(v)/len(v)) for k, v in epoch_train_losses.items() if len(v) > 0}
        if self.weighting_strategy and hasattr(self.weighting_strategy, 'update_epoch_losses'):
            self.weighting_strategy.update_epoch_losses(raw_means, epoch=epoch_index)
        aggregated = {f"train_{k}": v for k, v in raw_means.items()}
        return aggregated

    def _collect_weighted_task_losses(self, loss_dict: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        losses: List[torch.Tensor] = []
        if self.weighting_strategy is not None:
            for task in getattr(self.weighting_strategy, 'tasks', []):
                key = f'weighted_{task}_loss'
                val = loss_dict.get(key)
                if isinstance(val, torch.Tensor):
                    losses.append(val)
        return losses

    def _apply_pcgrad(self, loss_dict_or_tensor: Any) -> None:
        if not isinstance(loss_dict_or_tensor, dict):
            raise ValueError("PCGrad requires loss function to return a dict with weighted components")
        task_losses = self._collect_weighted_task_losses(loss_dict_or_tensor)
        if 'consistency_loss' in loss_dict_or_tensor:
            task_losses.append(self.loss_fn.w_consistency * loss_dict_or_tensor['consistency_loss'])
        if not task_losses:
            task_losses.append(loss_dict_or_tensor['total_loss'])
        params = [p for p in self.model.parameters() if p.requires_grad]
        task_grads = [
            torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            for loss in task_losses
        ]
        self.pcgrad.step(task_grads, params)

    @staticmethod
    def _tensor_dict_to_floats(tensors: Dict[str, torch.Tensor]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, tensor in tensors.items():
            if torch.is_tensor(tensor):
                out[key] = float(tensor.detach().item())
        return out

    def _compute_gradient_diagnostics(
        self, per_task_gradients: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        norm_stats: Dict[str, float] = {}
        cosine_stats: Dict[str, float] = {}
        tasks = list(per_task_gradients.keys())
        for task, grad in per_task_gradients.items():
            norm_stats[task] = float(grad.norm().detach().item())
        for idx, task_i in enumerate(tasks):
            for jdx in range(idx + 1, len(tasks)):
                task_j = tasks[jdx]
                gi = per_task_gradients[task_i]
                gj = per_task_gradients[task_j]
                denom = gi.norm() * gj.norm()
                if denom.detach().item() == 0:
                    cosine = 0.0
                else:
                    cosine = float(
                        F.cosine_similarity(gi.unsqueeze(0), gj.unsqueeze(0), dim=1).detach().item()
                    )
                cosine_stats[f'{task_i}_vs_{task_j}'] = cosine
        return norm_stats, cosine_stats

    def _extract_weighting_diagnostics(self) -> Dict[str, Any]:
        if not self.weighting_strategy:
            return {}
        getter = getattr(self.weighting_strategy, 'get_last_diagnostics', None)
        if callable(getter):
            latest = getter()
            return latest if latest else {}
        return {}

    def _maybe_record_loss_diagnostics(self, epoch_index: int, diagnostics: Dict[str, Any]) -> None:
        if not diagnostics:
            return
        if self.weighting_strategy:
            diagnostics.setdefault('strategy_type', self.weighting_strategy.__class__.__name__)
        else:
            diagnostics.setdefault('strategy_type', 'None')
        if self.global_step % max(1, self._log_frequency) != 0:
            return
        store_fn = getattr(self.metrics_storer, 'store_loss_diagnostics', None)
        if callable(store_fn):
            store_fn(step=self.global_step, epoch=epoch_index, diagnostics=diagnostics)

    def _validate_one_epoch(self, epoch: int = None) -> Dict[str, Any]:
        """
        Executes a single validation epoch using sliding window inference.
        It computes a full metrics report and dispatches data to both Tier 1 and Tier 2 systems.
        """
        self.model.eval()
        
        inferrer = SlidingWindowInferrer(
            model=self.model,
            patch_size_h=self.config.patch_size[0],
            patch_size_w=self.config.patch_size[1],
            stride_h=self.config.inference_stride[0],
            stride_w=self.config.inference_stride[1],
            device=self.device,
            batch_size=self.config.inference_batch_size
        )
        
        loop = tqdm(self.val_loader, desc=f"Validation", leave=False)
        epoch_val_losses = defaultdict(list)
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
                # Calculate validation loss (diagnostic only)
                # Build minimal masks_for_loss similarly to training for compatibility
                masks_for_loss = batch.get('masks', batch.get('mask'))
                if isinstance(masks_for_loss, dict):
                    masks_for_loss = {k: v.to(self.device, non_blocking=True) for k, v in masks_for_loss.items()}
                elif masks_for_loss is not None:
                    masks_for_loss = masks_for_loss.to(self.device, non_blocking=True)

                if masks_for_loss is not None:
                    loss_dict_or_tensor = self.loss_fn(predictions_for_metrics, masks_for_loss)
                    if isinstance(loss_dict_or_tensor, dict):
                        for key, val in loss_dict_or_tensor.items():
                            if torch.is_tensor(val):
                                epoch_val_losses[key].append(float(val.detach().item()))
                    else:
                        epoch_val_losses['total_loss'].append(float(loss_dict_or_tensor.detach().item()))

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
        metrics_report = self.metrics_calculator.compute()
        # Append averaged validation loss metrics under val_ namespace
        if epoch_val_losses:
            val_loss_means = {f"val_{k}": float(sum(v)/len(v)) for k, v in epoch_val_losses.items() if len(v) > 0}
            metrics_report.setdefault('loss_metrics', {}).update(val_loss_means)
        return metrics_report

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
                self.current_epoch = epoch + 1
                train_loss_report = self._train_one_epoch(epoch + 1)
                
                # Only validate if scheduled (saves significant time)
                if self._should_validate(epoch + 1):
                    # Reset metrics calculator at the start of each validation epoch
                    self.metrics_calculator.reset()
                    val_metrics_report = self._validate_one_epoch(epoch + 1)
                    # Merge train + val loss metrics into optimization_metrics like structure
                    optimization_metrics = val_metrics_report.get('optimization_metrics', {})
                    optimization_metrics.update(train_loss_report)
                    # Include validation loss metrics if present
                    if 'loss_metrics' in val_metrics_report:
                        optimization_metrics.update(val_metrics_report['loss_metrics'])
                    val_metrics_report['optimization_metrics'] = optimization_metrics
                    # Store history with combined metrics
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