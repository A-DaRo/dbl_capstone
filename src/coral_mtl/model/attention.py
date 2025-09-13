import torch
import torch.nn as nn
import torch.nn.functional as F

def _calculate_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float
) -> torch.Tensor:
    """A pure function to compute scaled dot-product attention."""
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attention_probs = F.softmax(attention_scores, dim=-1)
    return torch.matmul(attention_probs, value)


class MultiTaskCrossAttentionModule(nn.Module):
    """
    Implements the (legacy) symmetric Multi-Task Cross-Attention Module.
    This component uses OOP to manage the learnable projection layers (weights).
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.query_proj = nn.Linear(in_channels, in_channels)
        self.key_proj = nn.Linear(in_channels, in_channels)
        self.value_proj = nn.Linear(in_channels, in_channels)
        self.scale = in_channels ** -0.5

    def forward(
        self, F_genus: torch.Tensor, F_health: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs symmetric cross-attention.
        
        Args:
            F_genus: Fused feature tensor for the Genus task. Shape (B, C, H, W).
            F_health: Fused feature tensor for the Health task. Shape (B, C, H, W).

        Returns:
            A tuple of (Enriched_F_genus, Enriched_F_health).
        """
        B, C, H, W = F_genus.shape
        
        # Reshape for attention: (B, C, H, W) -> (B, H*W, C)
        F_genus_seq = F_genus.flatten(2).permute(0, 2, 1)
        F_health_seq = F_health.flatten(2).permute(0, 2, 1)

        # --- Part A: Health queries Genus ---
        enriched_F_health_seq = _calculate_attention(
            query=self.query_proj(F_health_seq),
            key=self.key_proj(F_genus_seq),
            value=self.value_proj(F_genus_seq),
            scale=self.scale
        )

        # --- Part B: Genus queries Health (Symmetric) ---
        enriched_F_genus_seq = _calculate_attention(
            query=self.query_proj(F_genus_seq),
            key=self.key_proj(F_health_seq),
            value=self.value_proj(F_health_seq),
            scale=self.scale
        )

        # Reshape back to image format: (B, H*W, C) -> (B, C, H, W)
        Enriched_F_health = enriched_F_health_seq.permute(0, 2, 1).reshape(B, C, H, W)
        Enriched_F_genus = enriched_F_genus_seq.permute(0, 2, 1).reshape(B, C, H, W)
        
        return Enriched_F_genus, Enriched_F_health