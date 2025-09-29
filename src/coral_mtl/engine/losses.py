import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, TYPE_CHECKING
import segmentation_models_pytorch as smp
from .loss_weighting import WeightingStrategy
import math

if TYPE_CHECKING:
    from ..utils.task_splitter import TaskSplitter

EPS = 1e-7  # global epsilon for numerical stability

# --- Baseline Loss Function ---

class CoralLoss(nn.Module):
    """
    A flexible, single-task hybrid loss function for baselining on the Coralscapes dataset.
    It combines a primary classification loss (Focal or Cross-Entropy) with Dice Loss.
    This is NOT a multi-task loss and does not use uncertainty weighting.
    """
    def __init__(self,
                 primary_loss_type: str = 'focal',
                 hybrid_alpha: float = 0.5,
                 focal_gamma: float = 2.0,
                 dice_smooth: float = 1.0,
                 class_weights: Optional[torch.Tensor] = None,
                 ignore_index: int = -100):
        """
        Args:
            primary_loss_type (str): The main classification loss. One of ['focal', 'cross_entropy'].
            hybrid_alpha (float): Weight for the primary loss. Dice Loss weight is (1 - alpha).
            focal_gamma (float): Focusing parameter for Focal Loss (used if primary_loss_type='focal').
            dice_smooth (float): Smoothing factor for Dice Loss.
            class_weights (torch.Tensor, optional): Optional class weights for Cross-Entropy.
            ignore_index (int): Specifies a target value to be ignored by all loss components.
        """
        super().__init__()
        
        if primary_loss_type not in ['focal', 'cross_entropy']:
            raise ValueError(f"primary_loss_type must be 'focal' or 'cross_entropy', but got {primary_loss_type}")
            
        self.hybrid_alpha = hybrid_alpha
        self.primary_loss_type = primary_loss_type
        self.ignore_index = ignore_index
        
        # Instantiate primary loss component based on the choice
        if primary_loss_type == 'focal':
            self.primary_loss = smp.losses.FocalLoss(
                mode='multiclass',
                gamma=focal_gamma,
                ignore_index=ignore_index
            )
            self.cross_entropy_fallback = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=ignore_index
            )
        elif primary_loss_type == 'cross_entropy':
            self.primary_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=ignore_index
            )
            self.cross_entropy_fallback = self.primary_loss
        
        # NOTE: we keep smp DiceLoss but wrap with additional epsilon guard in forward
        self.dice_loss = smp.losses.DiceLoss(
            mode='multiclass',
            ignore_index=ignore_index,
            smooth=dice_smooth
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Check for NaN inputs
        if torch.any(torch.isnan(logits)):
            print("Warning: NaN detected in CombinedLoss logits")
            return torch.tensor(1.0, device=logits.device, requires_grad=True)
        
        if torch.any(torch.isnan(targets)):
            print("Warning: NaN detected in CombinedLoss targets")
            return torch.tensor(1.0, device=logits.device, requires_grad=True)
            
        loss_primary = self.primary_loss(logits, targets.long())
        # Dice can return NaN if both prediction & target have no foreground; guard it
        loss_dice = self.dice_loss(logits, targets.long())
        if torch.isnan(loss_dice):
            # Re-compute a manual safe dice fallback (very rare)
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                targ = targets.long()
                valid_mask = (targ != self.ignore_index)
                if valid_mask.any():
                    one_hot = F.one_hot(torch.clamp(targ, min=0), num_classes=probs.shape[1]).permute(0,3,1,2).float()
                    one_hot = one_hot * valid_mask.unsqueeze(1)
                    intersect = (probs * one_hot).sum(dim=(0,2,3))
                    denom = probs.sum(dim=(0,2,3)) + one_hot.sum(dim=(0,2,3)) + EPS
                    manual_dice = 1 - (2*intersect / denom).mean()
                else:
                    manual_dice = torch.tensor(0.0, device=logits.device)
            loss_dice = manual_dice.detach().requires_grad_(True)
        
        # Check for NaN in loss components
        if torch.isnan(loss_primary):
            print("Warning: NaN detected in primary loss component; recomputing with cross-entropy fallback")
            loss_primary = self.cross_entropy_fallback(logits, targets.long())
            
        if torch.isnan(loss_dice):
            print("Warning: NaN detected in dice loss component")
            loss_dice = torch.tensor(0.5, device=logits.device, requires_grad=True)
        
        # Only add dice loss if its weight is > 0 to avoid unnecessary computation
        if self.hybrid_alpha < 1.0:
            total_loss = self.hybrid_alpha * loss_primary + (1 - self.hybrid_alpha) * loss_dice
        else:
            total_loss = loss_primary
            
        # Final NaN check
        if torch.isnan(total_loss):
            print("Warning: NaN detected in CombinedLoss total loss")
            total_loss = torch.tensor(1.0, device=logits.device, requires_grad=True)
            
        return total_loss

# --- Main Composite Loss Function ---

class CoralMTLLoss(nn.Module):
    """Multi-task loss orchestration delegating weighting to a strategy.

    Responsibilities post-refactor (Phase B):
      1. Compute raw per-task losses (primary: hybrid, auxiliary: CE).
      2. Apply optional logical consistency regularizer.
      3. Delegate aggregation / balancing to injected `WeightingStrategy`.
      4. Return comprehensive dict of components (unweighted + weighted + aux losses).
    """
    def __init__(self,
                 num_classes: Dict[str, int],
                 primary_tasks: List[str],
                 aux_tasks: List[str],
                 weighting_strategy: 'WeightingStrategy',
                 class_weights: Optional[Dict[str, torch.Tensor]] = None,
                 ignore_index: int = -100,
                 hybrid_alpha: float = 0.5,
                 focal_gamma: float = 2.0,
                 debug: bool = False,
                 splitter: Optional['TaskSplitter'] = None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.debug = debug
        self.weighting_strategy = weighting_strategy
        self.splitter = splitter
        self._running_stats = {
            'steps': 0,
            'per_task_loss_mean': {},
            'per_task_loss_m2': {}
        }

        self.primary_tasks = primary_tasks
        self.aux_tasks = aux_tasks
        self.primary_loss_fn = CoralLoss(
            hybrid_alpha=hybrid_alpha,
            focal_gamma=focal_gamma,
            ignore_index=self.ignore_index
        )
        class_weights = class_weights if class_weights is not None else {}
        self.aux_losses = nn.ModuleDict({
            task: nn.CrossEntropyLoss(
                weight=class_weights.get(task),
                ignore_index=self.ignore_index
            )
            for task in self.aux_tasks if task in self.num_classes
        })
        if self.debug:
            print(f"[CoralMTLLoss] Initialized (Strategy={self.weighting_strategy.__class__.__name__}) | primary={self.primary_tasks} aux={self.aux_tasks} ignore_index={ignore_index}")



    def _validate_task_predictions(self, task: str, predictions_tensor: torch.Tensor) -> bool:
        """
        Validate that prediction tensor has the correct number of channels for the task.
        
        Args:
            task (str): Task name
            predictions_tensor (torch.Tensor): Model predictions tensor (N, C, H, W)
            
        Returns:
            bool: True if valid, False if mismatch detected
        """
        expected_classes = self.num_classes.get(task)
        if expected_classes is None:
            if self.debug:
                print(f"[CoralMTLLoss] Task '{task}' not found in num_classes config -> skipping")
            return False
            
        actual_classes = predictions_tensor.shape[1]
        if actual_classes != expected_classes:
            print(f"[CoralMTLLoss] WARNING: Task '{task}' prediction shape mismatch - "
                  f"got {actual_classes} channels, expected {expected_classes} classes. Skipping task loss.")
            return False
            
        return True

    def _update_running_stats(self, loss_map: Dict[str, torch.Tensor]):
        if not self.debug:
            return
        self._running_stats['steps'] += 1
        for k, v in loss_map.items():
            if not k.startswith('unweighted_'):
                continue
            val = float(v.detach().item())
            if k not in self._running_stats['per_task_loss_mean']:
                self._running_stats['per_task_loss_mean'][k] = val
                self._running_stats['per_task_loss_m2'][k] = 0.0
            else:
                mean = self._running_stats['per_task_loss_mean'][k]
                m2 = self._running_stats['per_task_loss_m2'][k]
                n = self._running_stats['steps']
                delta = val - mean
                mean += delta / n
                m2 += delta * (val - mean)
                self._running_stats['per_task_loss_mean'][k] = mean
                self._running_stats['per_task_loss_m2'][k] = m2
        if self._running_stats['steps'] % 50 == 0:
            summary = {t: round(self._running_stats['per_task_loss_mean'][t],4) for t in self._running_stats['per_task_loss_mean']}
            print(f"[CoralMTLLoss][debug] running mean unweighted losses: {summary}")

    def compute_unweighted_losses(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute and return per-task unweighted losses (primary + auxiliary).

        This isolates raw task loss computation so gradient-based strategies
        can perform multiple backward passes externally.
        """
        unweighted_primary_losses: Dict[str, torch.Tensor] = {}
        for task in self.primary_tasks:
            if task in predictions and task in targets:
                unweighted_primary_losses[task] = self.primary_loss_fn(predictions[task], targets[task])
        unweighted_aux_losses: Dict[str, torch.Tensor] = {}
        for task in self.aux_tasks:
            if task in predictions and task in targets and task in self.aux_losses:
                unweighted_aux_losses[task] = self.aux_losses[task](predictions[task], targets[task].long())
        merged = {**unweighted_primary_losses, **unweighted_aux_losses}
        return merged

    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        loss_dict = {}
        
        # --- 1. Calculate Individual Task Losses ---
        unweighted_primary_losses = {}
        for task in self.primary_tasks:
            if task in predictions and task in targets:
                # Validate prediction tensor shape before computing loss
                if not self._validate_task_predictions(task, predictions[task]):
                    continue
                unweighted_primary_losses[task] = self.primary_loss_fn(predictions[task], targets[task])

        unweighted_aux_losses = {}
        for task in self.aux_tasks:
            if task in predictions and task in targets:
                # Validate prediction tensor shape before computing loss
                if not self._validate_task_predictions(task, predictions[task]):
                    continue
                aux_loss = self.aux_losses[task](predictions[task], targets[task].long())
                if torch.isnan(aux_loss):
                    if self.debug:
                        print(f"[CoralMTLLoss] NaN detected in auxiliary loss for task '{task}' -> replacing with 0")
                    aux_loss = torch.zeros_like(aux_loss)
                unweighted_aux_losses[task] = aux_loss
        
        if predictions:
            base_device = next(iter(predictions.values())).device
        elif targets:
            base_device = next(iter(targets.values())).device
        else:
            base_device = torch.device('cpu')

        loss_auxiliary_sum = sum(unweighted_aux_losses.values()) if unweighted_aux_losses else \
            torch.tensor(0.0, device=base_device)

        # --- 2. Build unified raw losses dict for strategy ---
        raw_losses: Dict[str, torch.Tensor] = {}
        raw_losses.update(unweighted_primary_losses)
        # Represent auxiliary as either individual tasks (preferred for strategy transparency)
        raw_losses.update(unweighted_aux_losses)
        if raw_losses:
            self.weighting_strategy.cache_unweighted_losses(raw_losses)
            weighted = self.weighting_strategy(raw_losses)
        else:
            weighted = {'total_loss': torch.zeros((), device=base_device)}
        
        if not torch.isfinite(weighted['total_loss']):
            if self.debug:
                print(f"[CoralMTLLoss] Non-finite total loss detected post-strategy (value={weighted['total_loss']}).")
            weighted['total_loss'] = torch.nan_to_num(weighted['total_loss'], nan=1.0, posinf=1.0, neginf=1.0)
        # --- 4. Aggregate primary/auxiliary balanced losses for reporting ---
        primary_terms = [weighted[f'weighted_{t}_loss'] for t in self.primary_tasks if f'weighted_{t}_loss' in weighted]
        aux_terms = [weighted[f'weighted_{t}_loss'] for t in self.aux_tasks if f'weighted_{t}_loss' in weighted]
        if primary_terms:
            weighted['primary_balanced_loss'] = sum(primary_terms)
        else:
            weighted['primary_balanced_loss'] = torch.zeros((), device=base_device)
        if aux_terms:
            weighted['aux_balanced_loss'] = sum(aux_terms)
        else:
            weighted['aux_balanced_loss'] = torch.zeros((), device=base_device)

        # --- 5. Compose final dict: unweighted_* + weighted outputs ---
        for task, loss in unweighted_primary_losses.items():
            weighted[f'unweighted_{task}_loss'] = loss
        for task, loss in unweighted_aux_losses.items():
            weighted[f'unweighted_{task}_loss'] = loss
        
        weighted['unweighted_aux_loss_sum'] = loss_auxiliary_sum
        
        # Running stats
        self._update_running_stats(weighted)
        
        return weighted