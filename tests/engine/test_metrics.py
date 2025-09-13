import torch
import pytest
from coral_mtl.engine.metrics import CoralMTLMetrics

@pytest.fixture
def metrics_setup():
    """Provides a metrics calculator instance and dummy data."""
    device = torch.device('cpu')
    num_classes = {
        'genus': 9, 'health': 4, 'fish': 2, 'human_artifacts': 2, 'substrate': 4
    }
    metrics_calc = CoralMTLMetrics(num_classes=num_classes, device=device)
    
    B, H, W = 2, 64, 64
    preds = {
        task: torch.randn(B, n_cls, H, W) for task, n_cls in num_classes.items()
    }
    # Create targets with a mix of perfect and random predictions for realistic metrics
    targets = {task: torch.argmax(logits, dim=1) for task, logits in preds.items()}
    for task, n_cls in num_classes.items():
        noise = torch.randint(0, n_cls, (B//2, H, W))
        targets[task][B//2:] = noise
        
    return metrics_calc, preds, targets

def test_metrics_calculator_update_compute_reset(metrics_setup):
    """
    Tests the full update -> compute -> reset cycle of the metrics calculator.
    """
    metrics_calc, preds, targets = metrics_setup
    
    # 1. Update with one batch
    metrics_calc.update(preds, targets)
    
    # Check if confusion matrices were updated
    assert torch.sum(metrics_calc.confusion_matrices['genus']) > 0
    assert metrics_calc.biou_stats['health']['union'] > 0
    
    # 2. Compute metrics
    final_metrics = metrics_calc.compute()
    
    assert isinstance(final_metrics, dict)
    expected_keys = ['H-Mean', 'mIoU_genus', 'BIoU_health', 'mPA_fish', 'IoU_substrate_class_1']
    for key in expected_keys:
        assert key in final_metrics
        assert isinstance(final_metrics[key], (float, int))
    
    # Check that mIoU is within a valid range
    assert 0.0 <= final_metrics['mIoU_genus'] <= 1.0

    # 3. Reset the calculator
    metrics_calc.reset()
    
    # Check if internal states are zeroed out
    assert torch.all(metrics_calc.confusion_matrices['genus'] == 0)
    assert metrics_calc.biou_stats['health']['intersection'] == 0.0
    assert metrics_calc.biou_stats['health']['union'] == 0.0