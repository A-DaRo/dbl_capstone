import torch
import pytest
from coral_mtl.model.core import CoralMTLModel, BaselineSegformer

def test_coral_mtl_model_end_to_end_smoke_test():
    """
    Performs a 'smoke test' on the fully assembled CoralMTLModel.
    It checks if the model can be instantiated and if a forward pass
    completes without errors, producing outputs of the correct shape and type.
    """
    B, H, W = 2, 256, 256
    
    num_classes = {
        'genus': 9,
        'health': 4,
        'fish': 2,
        'human_artifacts': 2,
        'substrate': 4
    }
    
    # Use a small, fast encoder for the test
    model = CoralMTLModel(
        encoder_name="nvidia/mit-b0",
        decoder_channel=128,
        num_classes=num_classes,
        attention_dim=64,
        primary_tasks=['genus', 'health'],
        aux_tasks=['fish', 'human_artifacts', 'substrate']
    )
    
    dummy_image = torch.randn(B, 3, H, W)
    
    with torch.no_grad():
        outputs = model(dummy_image)
        
    assert isinstance(outputs, dict)
    assert set(outputs.keys()) == set(num_classes.keys())
    
    for task_name, logits in outputs.items():
        assert logits.shape == (B, num_classes[task_name], H, W)
        assert logits.dtype == torch.float32

def test_baseline_segformer_smoke_test():
    """
    Performs a 'smoke test' on the BaselineSegformer model.
    """
    B, H, W = 2, 256, 256
    num_classes = 39  # Total flattened classes in Coralscapes
    
    model = BaselineSegformer(
        encoder_name="nvidia/mit-b0",
        decoder_channel=128,
        num_classes=num_classes
    )
    
    dummy_image = torch.randn(B, 3, H, W)
    
    with torch.no_grad():
        outputs = model(dummy_image)
        
    # Baseline model returns a single tensor, not a dictionary
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (B, num_classes, H, W)
    assert outputs.dtype == torch.float32