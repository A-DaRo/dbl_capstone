import json
from pathlib import Path
import torch
import pytest
from types import SimpleNamespace

from coral_mtl.metrics.metrics_storer import MetricsStorer
from coral_mtl.engine.losses import CoralMTLLoss
from coral_mtl.engine.loss_weighting import build_weighting_strategy


def test_save_test_loss_report(tmp_path):
    """Ensure atomic write and correct JSON content for test loss report."""
    storer = MetricsStorer(output_dir=str(tmp_path))
    losses = {"test_total_loss": 1.2345, "test_unweighted_genus_loss": 0.5}
    storer.save_test_loss_report(losses)
    path = tmp_path / 'test_loss_metrics.json'
    assert path.exists(), "test_loss_metrics.json not created"
    data = json.loads(path.read_text())
    assert data == losses


def test_history_integration_train_val_namespacing(tmp_path):
    """Simulate store_epoch_history with merged optimization metrics including train_/val_ prefixes."""
    storer = MetricsStorer(output_dir=str(tmp_path))
    report = {
        'optimization_metrics': {
            'H-Mean': 0.6,
            'train_total_loss': 2.0,
            'val_total_loss': 1.5
        }
    }
    storer.store_epoch_history(report, epoch=1)
    history_path = tmp_path / 'history.json'
    assert history_path.exists()
    history = json.loads(history_path.read_text())
    # Expect epoch list and metric keys present
    assert 1 in history['epoch']
    assert 'H-Mean' in history
    assert 'train_total_loss' in history
    assert 'val_total_loss' in history


@pytest.mark.parametrize("with_aux", [True, False])
def test_end_to_end_loss_and_test_report(tmp_path, device, with_aux):
    """Mini integration: run CoralMTLLoss on synthetic data and persist test report."""
    # Setup synthetic tasks
    num_classes = {'genus': 3, 'health': 2}
    if with_aux:
        num_classes['fish'] = 2
    primary = ['genus', 'health']
    aux = [t for t in num_classes if t not in primary]
    strategy = build_weighting_strategy({'type': 'uncertainty'}, primary, aux)
    loss_fn = CoralMTLLoss(num_classes, primary, aux, weighting_strategy=strategy, debug=True)

    b,h,w = 2,8,8
    predictions = {}
    targets = {}
    for task, n_cls in num_classes.items():
        logits = torch.randn(b, n_cls, h, w, device=device)
        target = torch.randint(0, n_cls, (b,h,w), device=device)
        predictions[task] = logits
        targets[task] = target

    loss_dict = loss_fn(predictions, targets)
    # Build namespaced test losses and persist
    storer = MetricsStorer(output_dir=str(tmp_path))
    test_losses = {f"test_{k}": float(v.detach().item()) for k,v in loss_dict.items() if torch.is_tensor(v)}
    storer.save_test_loss_report(test_losses)
    path = tmp_path / 'test_loss_metrics.json'
    content = json.loads(path.read_text())
    assert all(k.startswith('test_') for k in content.keys())
    assert 'test_total_loss' in content


def test_store_loss_diagnostics(tmp_path):
    storer = MetricsStorer(output_dir=str(tmp_path))
    storer.open_for_run(is_testing=False)
    diagnostics = {
        "strategy_type": "UncertaintyWeightingStrategy",
        "task_weights": {"genus": 0.6, "health": 0.4},
        "gradient_norm": {"genus": 1.0},
        "gradient_cosine_similarity": {"genus_vs_health": 0.25},
        "gradient_update_norm": 1.5
    }
    storer.store_loss_diagnostics(step=5, epoch=2, diagnostics=diagnostics)
    storer.close()

    diag_path = tmp_path / 'loss_diagnostics.jsonl'
    assert diag_path.exists()
    lines = diag_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record['step'] == 5
    assert record['epoch'] == 2
    assert record['task_weights']['genus'] == pytest.approx(0.6)
    assert record['gradient_norm']['genus'] == pytest.approx(1.0)
    assert record['gradient_cosine_similarity']['genus_vs_health'] == pytest.approx(0.25)
    assert record['gradient_update_norm'] == pytest.approx(1.5)