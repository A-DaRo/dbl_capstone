import torch
import pytest
from coral_mtl.model.attention import MultiTaskCrossAttentionModule

def test_attention_module_forward_pass_shapes():
    """
    Tests if the MultiTaskCrossAttentionModule processes input tensors
    and returns output tensors of the correct shape.
    """
    in_channels = 256
    B, H, W = 2, 16, 16
    
    attention_module = MultiTaskCrossAttentionModule(in_channels=in_channels)
    
    # Create dummy feature tensors
    f_genus = torch.randn(B, in_channels, H, W)
    f_health = torch.randn(B, in_channels, H, W)
    
    # Forward pass
    enriched_genus, enriched_health = attention_module(f_genus, f_health)
    
    # Assert output shapes are identical to input shapes
    assert enriched_genus.shape == f_genus.shape
    assert enriched_health.shape == f_health.shape