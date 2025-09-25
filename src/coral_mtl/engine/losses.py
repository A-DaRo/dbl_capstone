import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import segmentation_models_pytorch as smp

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
        
        # Instantiate Dice loss component
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
        loss_dice = self.dice_loss(logits, targets.long())
        
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
    """
    Implements the complete multi-task loss for the Coral-MTL project.
    Uses hierarchical uncertainty weighting to balance primary and auxiliary
    task groups, and includes an optional logical consistency penalty.
    """
    def __init__(self,
                 num_classes: Dict[str, int],
                 primary_tasks: List[str],
                 aux_tasks: List[str],
                 class_weights: Optional[Dict[str, torch.Tensor]] = None,
                 ignore_index: int = -100,
                 w_consistency: float = 0.1,
                 hybrid_alpha: float = 0.5,
                 focal_gamma: float = 2.0):
        super().__init__()
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.w_consistency = w_consistency
        
        # Store tasks dynamically from constructor arguments
        self.primary_tasks = primary_tasks
        self.aux_tasks = aux_tasks
        
        # --- Loss components using smp for robustness ---
        # Primary tasks use the CoralLoss hybrid function
        self.primary_loss_fn = CoralLoss(
            hybrid_alpha=hybrid_alpha,
            focal_gamma=focal_gamma,
            ignore_index=self.ignore_index
        )
        
        # Auxiliary tasks use standard weighted Cross-Entropy
        class_weights = class_weights if class_weights is not None else {}
        self.aux_losses = nn.ModuleDict({
            task: nn.CrossEntropyLoss(
                weight=class_weights.get(task),
                ignore_index=self.ignore_index
            )
            for task in self.aux_tasks if task in self.num_classes
        })
        
        # --- Learnable uncertainty parameters for hierarchical balancing ---
        # Dynamically create parameters for each primary task
        self.log_vars_primary = nn.ParameterDict({
            task: nn.Parameter(torch.tensor(0.0))
            for task in self.primary_tasks if task in self.num_classes
        })
        # Single parameter for the entire auxiliary group
        self.log_var_aux_group = nn.Parameter(torch.tensor(0.0))
        
        print(f"Initialized CoralMTLLoss with Primary Tasks: {self.primary_tasks}, "
              f"Auxiliary Tasks: {self.aux_tasks}, and ignore_index={ignore_index}")

    def _consistency_loss(self, genus_logits, health_logits):
        """Penalizes illogical predictions: P(healthy) > 0.5 AND P(genus=background) > 0.5"""
        # Add numerical stability checks
        if torch.any(torch.isnan(genus_logits)) or torch.any(torch.isnan(health_logits)):
            print("Warning: NaN detected in consistency loss inputs")
            return torch.tensor(0.0, device=genus_logits.device)
            
        genus_probs = F.softmax(genus_logits, dim=1)
        health_probs = F.softmax(health_logits, dim=1)
        
        # Check for NaN in probabilities
        if torch.any(torch.isnan(genus_probs)) or torch.any(torch.isnan(health_probs)):
            print("Warning: NaN detected in softmax probabilities")
            return torch.tensor(0.0, device=genus_logits.device)
        
        # Assuming class index 0 is background/ignored for genus task
        idx_genus_background = 0
        # Assuming class index 1 is 'alive/healthy' for health task
        idx_health_alive = 1 
        
        p_genus_background = genus_probs[:, idx_genus_background, :, :]
        p_health_alive = health_probs[:, idx_health_alive, :, :]
        
        illogical_mask = (p_health_alive > 0.5) & (p_genus_background > 0.5)
        
        if not illogical_mask.any():
            return torch.tensor(0.0, device=genus_logits.device)
            
        penalty = p_genus_background[illogical_mask].mean()
        
        # Final safety check for NaN
        if torch.isnan(penalty):
            print("Warning: NaN detected in consistency penalty")
            penalty = torch.tensor(0.0, device=genus_logits.device)
            
        return penalty

    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        loss_dict = {}
        
        # --- 1. Calculate Individual Task Losses ---
        unweighted_primary_losses = {}
        for task in self.primary_tasks:
            if task in predictions and task in targets:
                unweighted_primary_losses[task] = self.primary_loss_fn(predictions[task], targets[task])

        unweighted_aux_losses = {}
        for task in self.aux_tasks:
            if task in predictions and task in targets:
                unweighted_aux_losses[task] = self.aux_losses[task](predictions[task], targets[task].long())
        
        if predictions:
            base_device = next(iter(predictions.values())).device
        elif targets:
            base_device = next(iter(targets.values())).device
        else:
            base_device = torch.device('cpu')

        loss_auxiliary_sum = sum(unweighted_aux_losses.values()) if unweighted_aux_losses else \
            torch.tensor(0.0, device=base_device)

        # --- 2. Hierarchical Uncertainty Weighting with numerical stability ---
        loss_primary_balanced = torch.tensor(0.0, device=loss_auxiliary_sum.device)
        for task, loss in unweighted_primary_losses.items():
            log_var_clamped = torch.clamp(self.log_vars_primary[task], -10, 10)
            precision = torch.exp(-log_var_clamped)
            loss_primary_balanced += (precision * loss + 0.5 * log_var_clamped)
        
        log_var_aux_clamped = torch.clamp(self.log_var_aux_group, -10, 10)
        precision_aux_group = torch.exp(-log_var_aux_clamped)
        loss_auxiliary_balanced = precision_aux_group * loss_auxiliary_sum + 0.5 * log_var_aux_clamped
        
        # --- 3. Optional Consistency Regularizer ---
        loss_consistency = torch.tensor(0.0, device=loss_auxiliary_sum.device)
        if self.w_consistency > 0 and 'genus' in self.primary_tasks and 'health' in self.primary_tasks:
            if 'genus' in predictions and 'health' in predictions:
                loss_consistency = self._consistency_loss(predictions['genus'], predictions['health'])
                if torch.isnan(loss_consistency):
                    print(f"Warning: NaN detected in consistency loss: {loss_consistency}")
                    loss_consistency = torch.tensor(0.0, device=loss_auxiliary_sum.device)

        # --- 4. Calculate Final Total Loss ---
        total_loss = loss_primary_balanced + loss_auxiliary_balanced + self.w_consistency * loss_consistency
        
        # Final NaN check
        if torch.isnan(total_loss):
            print(f"Warning: NaN detected in total loss. Setting to 1.0 to prevent training collapse")
            total_loss = torch.tensor(1.0, device=loss_auxiliary_sum.device, requires_grad=True)

        # --- Populate dictionary of all components for logging ---
        loss_dict['total_loss'] = total_loss
        loss_dict['primary_balanced_loss'] = loss_primary_balanced
        loss_dict['aux_balanced_loss'] = loss_auxiliary_balanced
        loss_dict['consistency_loss'] = loss_consistency
        
        for task, loss in unweighted_primary_losses.items():
            loss_dict[f'unweighted_{task}_loss'] = loss
        
        loss_dict['unweighted_aux_loss_sum'] = loss_auxiliary_sum
        for task, loss in unweighted_aux_losses.items():
            loss_dict[f'unweighted_{task}_loss'] = loss

        for task, log_var in self.log_vars_primary.items():
            loss_dict[f'log_var_{task}'] = log_var.detach()
        loss_dict['log_var_aux_group'] = self.log_var_aux_group.detach()
        
        return loss_dict