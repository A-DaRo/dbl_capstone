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
                 ignore_index: int = 0):
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
        
        # Instantiate primary loss component based on the choice
        if primary_loss_type == 'focal':
            self.primary_loss = smp.losses.FocalLoss(
                mode='multiclass',
                gamma=focal_gamma,
                ignore_index=ignore_index
            )
        elif primary_loss_type == 'cross_entropy':
            self.primary_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=ignore_index
            )
        
        # Instantiate Dice loss component
        self.dice_loss = smp.losses.DiceLoss(
            mode='multiclass',
            ignore_index=ignore_index,
            smooth=dice_smooth
        )
        print(f"Initialized CoralLoss with primary loss '{primary_loss_type}' and ignore_index={ignore_index}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_primary = self.primary_loss(logits, targets.long())
        loss_dice = self.dice_loss(logits, targets.long())
        
        # Only add dice loss if its weight is > 0 to avoid unnecessary computation
        if self.hybrid_alpha < 1.0:
            total_loss = self.hybrid_alpha * loss_primary + (1 - self.hybrid_alpha) * loss_dice
        else:
            total_loss = loss_primary
            
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
                 class_weights: Optional[Dict[str, torch.Tensor]] = None,
                 ignore_index: int = 0, # Added for consistency
                 w_consistency: float = 0.1,
                 hybrid_alpha: float = 0.5,
                 focal_gamma: float = 2.0):
        super().__init__()
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.w_consistency = w_consistency
        
        self.primary_tasks = ['genus', 'health']
        self.aux_tasks = ['fish', 'human_artifacts', 'substrate']
        
        # --- Loss components using smp for robustness ---
        # Primary tasks use the CoralLoss hybrid function
        self.primary_loss_fn = CoralLoss(
            hybrid_alpha=hybrid_alpha,
            focal_gamma=focal_gamma,
            ignore_index=self.ignore_index # Propagate ignore_index
        )
        
        # Auxiliary tasks use standard weighted Cross-Entropy
        class_weights = class_weights if class_weights is not None else {}
        self.aux_losses = nn.ModuleDict({
            task: nn.CrossEntropyLoss(
                weight=class_weights.get(task),
                ignore_index=self.ignore_index
            )
            for task in self.aux_tasks
        })
        
        # --- Learnable uncertainty parameters for hierarchical balancing ---
        self.log_var_genus = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_var_health = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_var_aux_group = nn.Parameter(torch.zeros(1), requires_grad=True)
        print(f"Initialized CoralMTLLoss with Hierarchical Uncertainty Weighting and ignore_index={ignore_index}")

    def _consistency_loss(self, genus_logits, health_logits):
        """Penalizes illogical predictions: P(healthy) > 0.5 AND P(genus=background) > 0.5"""
        genus_probs = F.softmax(genus_logits, dim=1)
        health_probs = F.softmax(health_logits, dim=1)
        
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
        return penalty

    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # --- 1. Calculate Individual Task Losses ---
        loss_genus = self.primary_loss_fn(predictions['genus'], targets['genus'])
        loss_health = self.primary_loss_fn(predictions['health'], targets['health'])
        
        loss_aux_dict = {
            task: self.aux_losses[task](predictions[task], targets[task].long())
            for task in self.aux_tasks
        }
        loss_auxiliary_sum = sum(loss_aux_dict.values())
        
        # --- 2. Hierarchical Uncertainty Weighting ---
        precision_genus = torch.exp(-self.log_var_genus)
        precision_health = torch.exp(-self.log_var_health)
        loss_primary_balanced = (precision_genus * loss_genus + 0.5 * self.log_var_genus) + \
                                (precision_health * loss_health + 0.5 * self.log_var_health)
                                
        precision_aux_group = torch.exp(-self.log_var_aux_group)
        loss_auxiliary_balanced = precision_aux_group * loss_auxiliary_sum + 0.5 * self.log_var_aux_group
        
        # --- 3. Optional Consistency Regularizer ---
        loss_consistency = self._consistency_loss(predictions['genus'], predictions['health'])
        
        # --- 4. Calculate Final Total Loss ---
        total_loss = loss_primary_balanced + loss_auxiliary_balanced + self.w_consistency * loss_consistency

        # --- Return a dictionary of all components for logging ---
        loss_dict = {
            'total_loss': total_loss,
            'primary_balanced_loss': loss_primary_balanced,
            'aux_balanced_loss': loss_auxiliary_balanced,
            'consistency_loss': loss_consistency,
            'unweighted_genus_loss': loss_genus,
            'unweighted_health_loss': loss_health,
            'unweighted_aux_loss_sum': loss_auxiliary_sum,
            'log_var_genus': self.log_var_genus.detach(),
            'log_var_health': self.log_var_health.detach(),
            'log_var_aux_group': self.log_var_aux_group.detach()
        }
        loss_dict.update({f'unweighted_{task}_loss': loss for task, loss in loss_aux_dict.items()})
        
        return loss_dict


if __name__ == '__main__':
    # --- Sanity Check for CoralLoss (Baseline) ---
    print("\n--- Running Sanity Check for CoralLoss (Baseline) ---")
    B, C, H, W = 4, 39, 128, 128
    
    dummy_logits = torch.randn(B, C, H, W)
    dummy_targets_single_task = torch.randint(0, C, (B, H, W))
    
    baseline_loss_fn = CoralLoss(ignore_index=0)
    baseline_loss = baseline_loss_fn(dummy_logits, dummy_targets_single_task)
    
    print(f"CoralLoss (Baseline) output: {baseline_loss.item():.4f}")
    assert torch.is_tensor(baseline_loss) and baseline_loss.requires_grad
    print("Sanity check passed!")

    # --- Sanity Check for CoralMTLLoss ---
    print("\n--- Running Sanity Check for CoralMTLLoss ---")
    B, H, W = 2, 64, 64
    
    num_classes = {
        'genus': 9, 'health': 4, 'fish': 2, 'human_artifacts': 2, 'substrate': 4
    }
    
    dummy_predictions = {
        task: torch.randn(B, n_cls, H, W, requires_grad=True) for task, n_cls in num_classes.items()
    }
    dummy_targets = {
        task: torch.randint(0, C, (B, H, W)) for task, C in num_classes.items()
    }
    
    # Test with ignore_index = 0
    mtl_loss_fn_ignore = CoralMTLLoss(num_classes=num_classes, ignore_index=0)
    computed_losses = mtl_loss_fn_ignore(dummy_predictions, dummy_targets)
    
    print("\nComputed MTL Losses (with ignore_index=0):")
    for name, value in computed_losses.items():
        print(f"  - {name:<25}: {value.item():.4f}")
        
    computed_losses['total_loss'].backward()
    print("\nBackward pass successful.")
    
    print("\nAll sanity checks passed!")