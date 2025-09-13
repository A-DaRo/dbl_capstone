import torch
import torch.nn as nn
import pytest
from coral_mtl.engine.optimizer import create_optimizer_and_scheduler

class DummyModel(nn.Module):
    """A small model with different parameter types for testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3) # Has weight and bias
        self.ln = nn.LayerNorm(16) # Has weight and bias
        self.linear = nn.Linear(16, 10) # Has weight and bias
        # Note: BatchNorm weights are typically excluded from weight decay, but LayerNorm is more common in transformers.
        # The logic `param.dim() == 1` correctly handles biases and LayerNorm/BatchNorm weights.

def test_optimizer_parameter_grouping():
    """
    Verifies that create_optimizer_and_scheduler correctly separates parameters
    into 'decay' and 'no_decay' groups based on their name and dimension.
    """
    model = DummyModel()
    weight_decay_val = 0.01
    
    optimizer, _ = create_optimizer_and_scheduler(
        model=model,
        learning_rate=1e-4,
        weight_decay=weight_decay_val,
        num_training_steps=100,
        num_warmup_steps=10
    )
    
    param_groups = optimizer.param_groups
    
    assert len(param_groups) == 2, "Optimizer should have two parameter groups."
    
    decay_group = param_groups[0]
    no_decay_group = param_groups[1]
    
    # Check that weight decay is applied correctly
    assert decay_group['weight_decay'] == weight_decay_val
    assert no_decay_group['weight_decay'] == 0.0
    
    # Check parameter assignment
    decay_param_names = [name for name, _ in model.named_parameters() if 'weight' in name and 'ln' not in name]
    no_decay_param_names = [name for name, _ in model.named_parameters() if 'bias' in name or 'ln' in name]
    
    assert len(decay_group['params']) == len(decay_param_names)
    assert len(no_decay_group['params']) == len(no_decay_param_names)