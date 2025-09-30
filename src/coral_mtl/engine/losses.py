import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union, TYPE_CHECKING

import segmentation_models_pytorch as smp

from .loss_weighting import WeightingStrategy

if TYPE_CHECKING:
    from ..utils.task_splitter import TaskSplitter

EPS = 1e-7
BASELINE_TASK_KEY = "flattened"


class HybridSegmentationLoss(nn.Module):
    """Hybrid focal + dice loss used for individual segmentation tasks."""

    def __init__(
        self,
        hybrid_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1.0,
        ignore_index: int = 0,
    ) -> None:
        super().__init__()
        self.hybrid_alpha = float(hybrid_alpha)
        self.ignore_index = int(ignore_index)
        self.focal_loss = smp.losses.FocalLoss(
            mode="multiclass",
            gamma=float(focal_gamma),
            ignore_index=self.ignore_index,
        )
        self.cross_entropy_fallback = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass",
            ignore_index=self.ignore_index,
            smooth=float(dice_smooth),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        # first handle potential scenario of completely missing non-ignored targets
        targets_long = targets.long()
        valid_mask = (targets_long != self.ignore_index)
        if not valid_mask.any():
            # If there are no valid pixels in the target, the loss is zero
            # and should not contribute any gradients.
            print("Warning: No valid target pixels found; returning zero loss")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # habdle potential NaNs in inputs
        if torch.any(torch.isnan(logits)):
            print("Warning: NaN detected in HybridSegmentationLoss logits")
            return torch.tensor(1.0, device=logits.device, requires_grad=True)

        if torch.any(torch.isnan(targets)):
            print("Warning: NaN detected in HybridSegmentationLoss targets")
            return torch.tensor(1.0, device=logits.device, requires_grad=True)

        loss_primary = self.focal_loss(logits, targets_long)
        loss_dice = self.dice_loss(logits, targets_long)
        if torch.isnan(loss_dice):
            with torch.no_grad():
                probabilities = F.softmax(logits, dim=1)
                valid_mask = (targets_long != self.ignore_index)
                if valid_mask.any():
                    one_hot = F.one_hot(
                        torch.clamp(targets_long, min=0), num_classes=probabilities.shape[1]
                    ).permute(0, 3, 1, 2).float()
                    one_hot = one_hot * valid_mask.unsqueeze(1)
                    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
                    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3)) + EPS
                    manual_dice = 1 - (2 * intersection / denominator).mean()
                else:
                    manual_dice = torch.tensor(0.0, device=logits.device)
            loss_dice = manual_dice.detach().requires_grad_(True)

        if torch.isnan(loss_primary):
            print("Warning: NaN detected in focal component; recomputing with cross-entropy fallback")
            loss_primary = self.cross_entropy_fallback(logits, targets_long)

        if torch.isnan(loss_dice):
            print("Warning: NaN detected in dice component")
            loss_dice = torch.tensor(0.5, device=logits.device, requires_grad=True)

        if self.hybrid_alpha < 1.0:
            total_loss = self.hybrid_alpha * loss_primary + (1 - self.hybrid_alpha) * loss_dice
        else:
            total_loss = loss_primary

        if torch.isnan(total_loss):
            print("Warning: NaN detected in HybridSegmentationLoss total output")
            total_loss = torch.tensor(1.0, device=logits.device, requires_grad=True)

        return total_loss

class CoralLoss(nn.Module):
    """Unified loss orchestrator handling MTL and baseline configurations."""

    def __init__(
        self,
        per_task_loss_fns: nn.ModuleDict,
        weighting_strategy: Optional[WeightingStrategy],
        mode: str,
        *,
        clipping_config: Optional[Dict[str, Any]] = None,
        task_splitter: Optional["TaskSplitter"] = None,
    ) -> None:
        super().__init__()
        if mode not in {"mtl", "baseline"}:
            raise ValueError(f"Unsupported loss mode '{mode}'. Expected 'mtl' or 'baseline'.")
        if not per_task_loss_fns:
            raise ValueError("per_task_loss_fns must contain at least one task-specific loss module")

        self.per_task_loss_fns = per_task_loss_fns
        self.weighting_strategy = weighting_strategy
        self.mode = mode
        self.tasks = list(per_task_loss_fns.keys())
        self._baseline_task = self.tasks[0] if mode == "baseline" else None
        self.clipping_config = dict(clipping_config or {})
        self.task_splitter = task_splitter

    def _standardize_inputs(
        self,
        predictions: Union[Dict[str, torch.Tensor], torch.Tensor],
        targets: Union[Dict[str, torch.Tensor], torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if self.mode == "baseline":
            assert self._baseline_task is not None
            if isinstance(predictions, dict):
                prediction_tensor = next(iter(predictions.values())) if predictions else None
            else:
                prediction_tensor = predictions
            if isinstance(targets, dict):
                target_tensor = next(iter(targets.values())) if targets else None
            else:
                target_tensor = targets
            pred_dict = {self._baseline_task: prediction_tensor} if prediction_tensor is not None else {}
            target_dict = {self._baseline_task: target_tensor} if target_tensor is not None else {}
        else:
            if not isinstance(predictions, dict) or not isinstance(targets, dict):
                raise TypeError("MTL mode requires dictionary inputs for predictions and targets")
            pred_dict = {task: predictions[task] for task in self.tasks if task in predictions}
            target_dict = {task: targets[task] for task in self.tasks if task in targets}
        return pred_dict, target_dict

    def _compute_unweighted_from_dict(
        self,
        standardized_predictions: Dict[str, torch.Tensor],
        standardized_targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        unweighted: Dict[str, torch.Tensor] = {}
        for task, loss_module in self.per_task_loss_fns.items():
            prediction = standardized_predictions.get(task)
            target = standardized_targets.get(task)
            if prediction is None or target is None:
                continue
            unweighted[task] = loss_module(prediction, target)
        return unweighted

    @staticmethod
    def _infer_device(
        predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.device:
        if predictions:
            return next(iter(predictions.values())).device
        if targets:
            return next(iter(targets.values())).device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_unweighted_losses(
        self,
        predictions: Union[Dict[str, torch.Tensor], torch.Tensor],
        targets: Union[Dict[str, torch.Tensor], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        pred_dict, target_dict = self._standardize_inputs(predictions, targets)
        return self._compute_unweighted_from_dict(pred_dict, target_dict)

    def forward(
        self,
        predictions: Union[Dict[str, torch.Tensor], torch.Tensor],
        targets: Union[Dict[str, torch.Tensor], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        pred_dict, target_dict = self._standardize_inputs(predictions, targets)
        unweighted = self._compute_unweighted_from_dict(pred_dict, target_dict)
        device = self._infer_device(pred_dict, target_dict)

        if self.weighting_strategy is not None and unweighted:
            weighted = self.weighting_strategy(unweighted)
        elif unweighted:
            total = sum(unweighted.values())
            weighted = {"total_loss": total}
        else:
            weighted = {"total_loss": torch.zeros((), device=device)}

        for task_name, loss_value in unweighted.items():
            weighted[f"unweighted_{task_name}_loss"] = loss_value

        return weighted

    def get_clipping_parameters(self) -> Dict[str, Any]:
        return dict(self.clipping_config)


def build_loss(
    splitter: "TaskSplitter",
    weighting_strategy: Optional[WeightingStrategy],
    loss_config: Dict[str, Any],
) -> CoralLoss:
    from ..utils.task_splitter import BaseTaskSplitter, MTLTaskSplitter

    params = (loss_config or {}).get("params", {})
    default_alpha = params.get("hybrid_alpha", 0.5)
    default_gamma = params.get("focal_gamma", 2.0)
    dice_smooth = params.get("dice_smooth", 1.0)
    ignore_index = params.get("ignore_index", 0)
    clipping_config = params.get("gradient_clipping")
    if clipping_config is not None and not isinstance(clipping_config, dict):
        raise ValueError("loss.params.gradient_clipping must be a mapping when provided")

    task_specific: Dict[str, Dict[str, Any]] = params.get("task_specific", {})

    if isinstance(splitter, MTLTaskSplitter):
        mode = "mtl"
        available_tasks = list(splitter.hierarchical_definitions.keys())
        if weighting_strategy is not None and getattr(weighting_strategy, 'tasks', None):
            task_names = [task for task in weighting_strategy.tasks if task in available_tasks]
        else:
            task_names = available_tasks
    elif isinstance(splitter, BaseTaskSplitter):
        mode = "baseline"
        task_names = [BASELINE_TASK_KEY]
    else:
        raise TypeError("Unsupported splitter type for build_loss")

    per_task_losses = nn.ModuleDict()
    for task_name in task_names:
        overrides = task_specific.get(task_name, {})
        per_task_losses[task_name] = HybridSegmentationLoss(
            hybrid_alpha=overrides.get("hybrid_alpha", default_alpha),
            focal_gamma=overrides.get("focal_gamma", default_gamma),
            dice_smooth=overrides.get("dice_smooth", dice_smooth),
            ignore_index=overrides.get("ignore_index", ignore_index),
        )

    return CoralLoss(
        per_task_loss_fns=per_task_losses,
        weighting_strategy=weighting_strategy,
        mode=mode,
        clipping_config=clipping_config,
        task_splitter=splitter,
    )