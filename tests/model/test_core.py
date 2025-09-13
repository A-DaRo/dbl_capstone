import torch
import pytest
from coral_mtl.model.core import CoralMTLModel

def test_coral_mtl_model_end_to_end_smoke_test():
    """
    Performs a 'smoke test' on the fully assembled CoralMTLModel.
    It checks if the model can be instantiated and if a forward pass
    completes without errors, producing outputs of the correct shape and type.
    """
    B, H, W = 2, 256, 256
    
    num_classes = {
        'panoptic_shape': 9,
        'panoptic_health': 4,
        'fish': 2,
        'human_artifacts': 2,
        'substrate': 4
    }
    
    # Use a small, fast encoder for the test
    model = CoralMTLModel(
        encoder_name="nvidia/mit-b0",
        decoder_channel=128,
        num_classes=num_classes,
        attention_dim=64
    )
    
    dummy_image = torch.randn(B, 3, H, W)
    
    with torch.no_grad():
        outputs = model(dummy_image)
        
    assert isinstance(outputs, dict)
    assert set(outputs.keys()) == set(num_classes.keys())
    
    for task_name, logits in outputs.items():
        assert logits.shape == (B, num_classes[task_name], H, W)
        assert logits.dtype == torch.float32