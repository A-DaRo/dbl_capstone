"""Tests for the :class:`CoralLoss` orchestrator in multi-task mode."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from coral_mtl.engine.losses import CoralLoss, build_loss
from coral_mtl.engine.loss_weighting import WeightingStrategy, build_weighting_strategy


def _task_num_classes(splitter, task: str) -> int:
    definition = splitter.hierarchical_definitions[task]
    if definition.get("is_grouped", False):
        return len(definition["grouped"]["id2label"])
    return len(definition["ungrouped"]["id2label"])


def _make_predictions_targets(splitter, tasks, device, *, good: bool) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    preds: dict[str, torch.Tensor] = {}
    targs: dict[str, torch.Tensor] = {}
    batch, size = 2, 8
    for task in tasks:
        n_cls = _task_num_classes(splitter, task)
        if n_cls < 2:
            continue
        logits = torch.full((batch, n_cls, size, size), -10.0, device=device)
        logits[:, 1] = 10.0 if good else -10.0
        if not good:
            logits[:, 0] = 10.0
        target_value = 1
        preds[task] = logits
        targs[task] = torch.full((batch, size, size), target_value, dtype=torch.long, device=device)
    return preds, targs


def _build_default_loss(splitter, device, *, clipping: bool = False) -> CoralLoss:
    tasks = list(splitter.hierarchical_definitions.keys())
    primary = [task for task in ("genus", "health") if task in tasks]
    auxiliary = [task for task in tasks if task not in primary]
    strategy = build_weighting_strategy(config=None, primary=primary, auxiliary=auxiliary)
    loss_config = {
        "params": {
            "ignore_index": -100,
            "gradient_clipping": {"max_norm": 1.0} if clipping else None,
        }
    }
    return build_loss(splitter, strategy, loss_config)


def test_perfect_predictions_low_loss(splitter_mtl, device):
    loss = _build_default_loss(splitter_mtl, device)
    preds, targets = _make_predictions_targets(splitter_mtl, loss.tasks, device, good=True)
    if not preds:
        pytest.skip("No tasks with more than one class available")
    result = loss(preds, targets)
    assert torch.isfinite(result["total_loss"])
    assert result["total_loss"].item() < 0.2


def test_wrong_predictions_high_loss(splitter_mtl, device):
    loss = _build_default_loss(splitter_mtl, device)
    preds, targets = _make_predictions_targets(splitter_mtl, loss.tasks, device, good=False)
    if not preds:
        pytest.skip("No tasks with more than one class available")
    result = loss(preds, targets)
    assert torch.isfinite(result["total_loss"])
    assert result["total_loss"].item() > 0.5


def test_missing_task_predictions_are_ignored(splitter_mtl, device):
    loss = _build_default_loss(splitter_mtl, device)
    preds, targets = _make_predictions_targets(splitter_mtl, loss.tasks, device, good=True)
    if not preds:
        pytest.skip("No tasks with more than one class available")
    # Drop one prediction entry intentionally
    dropped_task = next(iter(preds.keys()))
    preds.pop(dropped_task)
    unweighted = loss.compute_unweighted_losses(preds, targets)
    assert dropped_task not in unweighted
    assert all(torch.isfinite(val) for val in unweighted.values())


def test_weighting_strategy_receives_losses(splitter_mtl, device):
    tasks = list(splitter_mtl.hierarchical_definitions.keys())
    if not {"genus", "health"}.issubset(tasks):
        pytest.skip("Requires genus and health tasks")

    class RecordingStrategy(WeightingStrategy):
        def __init__(self, task_list):
            super().__init__(task_list)
            self.forward_invocations: list[dict[str, torch.Tensor]] = []

        def forward(self, unweighted_losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            self.forward_invocations.append(unweighted_losses)
            device = next(iter(unweighted_losses.values())).device
            total = torch.zeros((), device=device)
            for loss in unweighted_losses.values():
                total = total + 0.5 * loss
            return {
                "total_loss": total,
                **{f"weighted_{task}_loss": 0.5 * value for task, value in unweighted_losses.items()},
            }

    strategy = RecordingStrategy(["genus", "health"])
    loss = build_loss(splitter_mtl, strategy, {"params": {"ignore_index": -100}})
    preds, targets = _make_predictions_targets(splitter_mtl, ["genus", "health"], device, good=False)
    result = loss(preds, targets)
    assert strategy.forward_invocations, "Strategy.forward should be called"
    assert torch.isfinite(result["total_loss"])
    for task in strategy.tasks:
        assert f"unweighted_{task}_loss" in result
        assert f"weighted_{task}_loss" in result


def test_gradient_clipping_config_exposed(splitter_mtl, device):
    loss = _build_default_loss(splitter_mtl, device, clipping=True)
    clip_cfg = loss.get_clipping_parameters()
    assert clip_cfg == {"max_norm": 1.0}


def test_loss_handles_ignore_index(splitter_mtl, device):
    loss = _build_default_loss(splitter_mtl, device)
    task = next(iter(loss.tasks))
    n_cls = _task_num_classes(splitter_mtl, task)
    logits = torch.randn(1, n_cls, 4, 4, device=device)
    target = torch.full((1, 4, 4), -100, dtype=torch.long, device=device)
    result = loss({task: logits}, {task: target})
    assert torch.isfinite(result["total_loss"])


def test_toy_mtl_overfit_reduces_loss(splitter_mtl, device):
    loss = _build_default_loss(splitter_mtl, device)
    if not loss.tasks:
        pytest.skip("No tasks available")

    class TinyModel(nn.Module):
        def __init__(self, task_channels):
            super().__init__()
            self.heads = nn.ModuleDict({
                task: nn.Conv2d(3, channels, kernel_size=1)
                for task, channels in task_channels.items()
            })

        def forward(self, x):
            return {task: head(x) for task, head in self.heads.items()}

    task_channels = {
        task: _task_num_classes(splitter_mtl, task)
        for task in loss.tasks
    }
    if not all(channels >= 1 for channels in task_channels.values()):
        pytest.skip("Invalid task channel configuration")

    model = TinyModel(task_channels).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(loss.parameters()), lr=0.1)
    image = torch.randn(2, 3, 8, 8, device=device)
    targets = {
        task: torch.randint(0, channels, (2, 8, 8), device=device)
        for task, channels in task_channels.items()
    }

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        preds = model(image)
        result = loss(preds, targets)
        total = result["total_loss"]
        total.backward()
        optimizer.step()
        losses.append(float(total.detach().item()))

    assert all(later <= earlier for earlier, later in zip(losses, losses[1:]))


@pytest.mark.parametrize("fill_value", [-100, 0])
def test_nan_resilience(splitter_mtl, device, fill_value):
    loss = _build_default_loss(splitter_mtl, device)
    preds = {
        task: torch.zeros((1, _task_num_classes(splitter_mtl, task), 4, 4), device=device)
        for task in loss.tasks
    }
    targets = {
        task: torch.full((1, 4, 4), fill_value, dtype=torch.long, device=device)
        for task in loss.tasks
    }
    result = loss(preds, targets)
    assert torch.isfinite(result["total_loss"])
    for key, value in result.items():
        if torch.is_tensor(value):
            assert torch.isfinite(value)