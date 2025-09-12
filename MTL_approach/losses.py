import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

# --- Component Loss Functions ---

class DiceLoss(nn.Module):
    """
    Computes the Dice Loss, which is a popular metric for semantic segmentation
    that directly optimizes for spatial overlap (IoU).
    """
    def __init__(self, smooth: float = 1e-6, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw model output of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth labels of shape (B, H, W).
        """
        # Get probabilities from logits
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]

        # Create one-hot encoded target
        one_hot_targets = F.one_hot(targets.long(), num_classes=num_classes)
        one_hot_targets = one_hot_targets.permute(0, 3, 1, 2).contiguous() # (B, C, H, W)

        # Handle ignore_index by creating a mask
        mask = (targets != self.ignore_index).unsqueeze(1) # (B, 1, H, W)

        # Apply mask
        probs = probs * mask
        one_hot_targets = one_hot_targets * mask

        # Calculate intersection and union
        intersection = torch.sum(probs * one_hot_targets, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(one_hot_targets, dim=(2, 3))

        # Calculate Dice score per class, then average
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1. - dice_score.mean()
        
        return dice_loss

class FocalLoss(nn.Module):
    """
    Computes the Focal Loss, which addresses class imbalance by down-weighting
    the loss assigned to well-classified examples.
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 ignore_index: int = 255, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw model output of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth labels of shape (B, H, W).
        """
        # Calculate Cross-Entropy loss without reduction
        ce_loss = F.cross_entropy(logits, targets.long(), ignore_index=self.ignore_index, reduction='none')

        # Get probabilities of the correct class
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.long().unsqueeze(1)).squeeze(1)

        # Calculate the modulating factor
        modulating_factor = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets.long().flatten()).reshape(targets.shape)
            focal_loss = alpha_t * modulating_factor * ce_loss
        else:
            focal_loss = modulating_factor * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# --- Main Composite Loss Function ---

class CoralMTLLoss(nn.Module):
    """
    Implements the complete multi-task loss for the Coral-MTL project.
    Combines primary and auxiliary losses with uncertainty weighting and a
    logical consistency penalty.
    """
    def __init__(self,
                 num_classes: Dict[str, int],
                 class_weights: Optional[Dict[str, torch.Tensor]] = None,
                 w_aux: float = 0.4,
                 w_consistency: float = 0.2,
                 hybrid_alpha: float = 0.5,
                 focal_gamma: float = 2.0):
        super().__init__()
        
        self.num_classes = num_classes
        self.class_weights = class_weights if class_weights is not None else {}
        self.w_aux = w_aux
        self.w_consistency = w_consistency
        self.hybrid_alpha = hybrid_alpha
        
        self.primary_tasks = ['genus', 'health']
        self.aux_tasks = ['fish', 'human_artifacts', 'substrate']
        
        # --- Loss components ---
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        
        # Weighted Cross-Entropy for auxiliary tasks
        self.aux_losses = nn.ModuleDict({
            task: nn.CrossEntropyLoss(weight=self.class_weights.get(task))
            for task in self.aux_tasks
        })
        
        # --- Learnable uncertainty parameters (log variance) for primary tasks ---
        # Learning log(sigma^2) is more numerically stable than learning sigma directly.
        self.log_var_genus = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_var_health = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _hybrid_loss(self, logits, targets, task_name):
        """Calculates L_task = α * L_Focal + (1-α) * L_Dice"""
        # Note: FocalLoss can internally use class weights, but for simplicity
        # in this hybrid setup, we let it handle the hard examples via gamma.
        loss_focal = self.focal_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        return self.hybrid_alpha * loss_focal + (1 - self.hybrid_alpha) * loss_dice

    def _consistency_loss(self, genus_logits, health_logits):
        """Penalizes illogical predictions: P(healthy) > 0.5 AND P(genus=background) > 0.5"""
        genus_probs = F.softmax(genus_logits, dim=1)
        health_probs = F.softmax(health_logits, dim=1)
        
        # Class indices: 0 for genus background, 1 for 'alive' health state
        p_genus_background = genus_probs[:, 0, :, :]
        p_health_alive = health_probs[:, 1, :, :]
        
        # Identify illogical pixels
        illogical_mask = (p_health_alive > 0.5) & (p_genus_background > 0.5)
        
        if not illogical_mask.any():
            return torch.tensor(0.0, device=genus_logits.device)
            
        # The penalty is to push the genus prediction away from 'background'
        # We can use a simple penalty like the probability of the illogical state
        penalty = p_genus_background[illogical_mask].mean()
        
        return penalty

    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # --- 1. Calculate Primary Task Losses (L_genus, L_health) ---
        loss_genus = self._hybrid_loss(predictions['genus'], targets['genus'], 'genus')
        loss_health = self._hybrid_loss(predictions['health'], targets['health'], 'health')
        
        # --- 2. Calculate Auxiliary Task Losses ---
        loss_aux_dict = {
            task: self.aux_losses[task](predictions[task], targets[task].long())
            for task in self.aux_tasks
        }
        loss_auxiliary = sum(loss_aux_dict.values())
        
        # --- 3. Calculate Primary Loss with Uncertainty Weighting ---
        # Formula: L_primary = (1/σ_g²) * L_g + (1/σ_h²) * L_h + log(σ_g * σ_h)
        # Using log variance trick: L = exp(-log_var) * L_task + 0.5 * log_var
        precision_genus = torch.exp(-self.log_var_genus)
        precision_health = torch.exp(-self.log_var_health)
        
        loss_primary = (precision_genus * loss_genus + self.log_var_genus) + \
                       (precision_health * loss_health + self.log_var_health)
        
        # --- 4. Calculate Consistency Loss ---
        loss_consistency = self._consistency_loss(predictions['genus'], predictions['health'])
        
        # --- 5. Calculate Total Loss ---
        total_loss = loss_primary + self.w_aux * loss_auxiliary + self.w_consistency * loss_consistency

        # --- Return dictionary of all loss components for logging ---
        loss_dict = {
            'total_loss': total_loss,
            'primary_loss': loss_primary,
            'auxiliary_loss': loss_auxiliary,
            'consistency_loss': loss_consistency,
            'genus_loss': loss_genus,
            'health_loss': loss_health,
            'log_var_genus': self.log_var_genus.detach(),
            'log_var_health': self.log_var_health.detach()
        }
        loss_dict.update({f'{task}_loss': loss for task, loss in loss_aux_dict.items()})
        
        return loss_dict


if __name__ == '__main__':
    print("--- Running Sanity Check for CoralMTLLoss ---")
    B, H, W = 2, 64, 64
    
    num_classes = {
        'genus': 9, 'health': 4, 'fish': 2, 'human_artifacts': 2, 'substrate': 4
    }
    
    # --- Dummy Data ---
    dummy_predictions = {
        task: torch.randn(B, n_cls, H, W, requires_grad=True) for task, n_cls in num_classes.items()
    }
    dummy_targets = {
        task: torch.randint(0, n_cls, (B, H, W)) for task, n_cls in num_classes.items()
    }
    # Make some targets illogical for testing consistency loss
    dummy_targets['genus'][0, 10:20, 10:20] = 0 # background
    dummy_targets['health'][0, 10:20, 10:20] = 1 # alive/healthy
    
    # --- Instantiate Loss ---
    loss_fn = CoralMTLLoss(num_classes=num_classes, w_aux=0.4, hybrid_alpha=0.5)

    # --- Compute Loss ---
    computed_losses = loss_fn(dummy_predictions, dummy_targets)

    # --- Print Results ---
    print("\nComputed Losses:")
    for name, value in computed_losses.items():
        print(f"  - {name:<20}: {value.item():.4f}")
        
        # --- FIX: Corrected Assertion Logic ---
        assert torch.is_tensor(value)
        
        # Detached log variances for logging should NOT require a gradient.
        if 'log_var' in name:
            assert not value.requires_grad, f"'{name}' should be detached and not require a grad."
        # All other loss components are part of the computation graph and MUST require a gradient.
        else:
            assert value.requires_grad, f"'{name}' is part of the graph and should require a grad."

    print("\nSanity check passed! Loss function is working as expected.")