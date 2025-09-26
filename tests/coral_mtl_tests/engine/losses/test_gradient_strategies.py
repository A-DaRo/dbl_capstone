from typing import Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from coral_mtl.engine.gradient_strategies import (
    IMGradStrategy,
    NashMTLStrategy,
    _SOLVERS_AVAILABLE,
)

TASKS: List[str] = ["genus", "health"]


class _TinySharedNet(nn.Module):
    def __init__(self, tasks: List[str], num_classes: Dict[str, int]) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 1),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({task: nn.Conv2d(8, num_classes[task], 1) for task in tasks})

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.shared(x)
        return {task: self.heads[task](shared) for task in self.heads}


@pytest.fixture
def tiny_mtl_setup(device: torch.device) -> tuple[_TinySharedNet, List[str], Dict[str, int]]:
    tasks = TASKS
    num_classes = {"genus": 4, "health": 3}
    model = _TinySharedNet(tasks, num_classes).to(device)
    return model, tasks, num_classes


@pytest.fixture
def conflicting_gradients(device: torch.device) -> Dict[str, torch.Tensor]:
    grad_genus = torch.tensor([0.8, -0.2, 0.6, -0.1], device=device, dtype=torch.float32)
    grad_health = torch.tensor([-0.4, 0.9, -0.3, 0.5], device=device, dtype=torch.float32)
    return {"genus": grad_genus, "health": grad_health}


def _random_batch(tasks: List[str], num_classes: Dict[str, int], device: torch.device) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    x = torch.randn(2, 3, 8, 8, device=device)
    targets = {
        task: torch.randint(0, num_classes[task], (2, 8, 8), device=device)
        for task in tasks
    }
    return x, targets


def _collect_per_task_gradients(
    model: nn.Module,
    losses: Dict[str, torch.Tensor],
    params: List[torch.nn.Parameter],
    task_order: List[str],
) -> Dict[str, torch.Tensor]:
    gradients: Dict[str, torch.Tensor] = {}
    ordered_tasks = [task for task in task_order if task in losses]
    for idx, task in enumerate(ordered_tasks):
        model.zero_grad(set_to_none=True)
        retain_graph = idx < len(ordered_tasks) - 1
        losses[task].backward(retain_graph=retain_graph)
        grads_flat: List[torch.Tensor] = []
        for param in params:
            if param.grad is None:
                grads_flat.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
            else:
                grads_flat.append(param.grad.detach().clone().reshape(-1))
        gradients[task] = torch.cat(grads_flat)
    model.zero_grad(set_to_none=True)
    return gradients


def test_imgrad_pgd_fallback_used(monkeypatch, conflicting_gradients: Dict[str, torch.Tensor]) -> None:
    call_counter = {"pgd": 0}

    def fake_pgd(self, gradients: torch.Tensor) -> torch.Tensor:  # pragma: no cover - monkeypatched
        call_counter["pgd"] += 1
        task_count = gradients.shape[0]
        return torch.full((task_count,), 1.0 / task_count, device=gradients.device, dtype=gradients.dtype)

    monkeypatch.setattr(IMGradStrategy, "_mgda_weights_pgd", fake_pgd)
    strategy = IMGradStrategy(TASKS, solver="pgd")
    update = strategy.compute_update_vector(conflicting_gradients)
    assert call_counter["pgd"] == 1
    assert torch.isfinite(update).all()


def test_imgrad_auto_solver_respects_availability(monkeypatch, conflicting_gradients: Dict[str, torch.Tensor]) -> None:
    monkeypatch.setitem(_SOLVERS_AVAILABLE, "cvxopt", False)
    call_counter = {"pgd": 0}

    def fake_pgd(self, gradients: torch.Tensor) -> torch.Tensor:  # pragma: no cover - monkeypatched
        call_counter["pgd"] += 1
        task_count = gradients.shape[0]
        return torch.full((task_count,), 1.0 / task_count, device=gradients.device, dtype=gradients.dtype)

    monkeypatch.setattr(IMGradStrategy, "_mgda_weights_pgd", fake_pgd)
    strategy = IMGradStrategy(TASKS, solver="auto")
    update = strategy.compute_update_vector(conflicting_gradients)
    assert call_counter["pgd"] == 1
    assert not strategy.use_qp
    assert torch.isfinite(update).all()


def test_imgrad_blending_interpolates_between_mean_and_mgda(monkeypatch, conflicting_gradients: Dict[str, torch.Tensor]) -> None:
    weights = torch.tensor([0.7, 0.3], dtype=torch.float32)

    def fake_weights(self, gradients: torch.Tensor) -> torch.Tensor:  # pragma: no cover - monkeypatched
        return weights.to(device=gradients.device, dtype=gradients.dtype)

    monkeypatch.setattr(IMGradStrategy, "_mgda_weights_pgd", fake_weights)
    strategy = IMGradStrategy(TASKS, solver="pgd")
    update = strategy.compute_update_vector(conflicting_gradients)

    G = torch.stack([conflicting_gradients[t] for t in TASKS], dim=0)
    gm = (weights.unsqueeze(1) * G).sum(dim=0)
    g0 = G.mean(dim=0)
    gm_norm = torch.norm(gm)
    g0_norm = torch.norm(g0)
    if gm_norm <= 1e-12 or g0_norm <= 1e-12:
        expected = g0
        alpha = 0.0
    else:
        cos_theta = torch.clamp(torch.dot(g0, gm) / (g0_norm * gm_norm), -1.0, 1.0)
        alpha = 0.5 * (1 - cos_theta)
        expected = (1 - alpha) * g0 + alpha * gm

    assert torch.allclose(update, expected, atol=1e-6, rtol=1e-5)
    assert 0.0 <= alpha <= 1.0


@pytest.mark.optdeps
@pytest.mark.skipif(not _SOLVERS_AVAILABLE["cvxopt"], reason="cvxopt optional dependency not installed")
def test_imgrad_qp_solver_invoked_when_available(monkeypatch, conflicting_gradients: Dict[str, torch.Tensor]) -> None:
    call_counter = {"qp": 0}

    def fake_qp(self, gradients: torch.Tensor) -> torch.Tensor:  # pragma: no cover - monkeypatched
        call_counter["qp"] += 1
        task_count = gradients.shape[0]
        return torch.full((task_count,), 1.0 / task_count, device=gradients.device, dtype=gradients.dtype)

    def fail_pgd(self, gradients: torch.Tensor) -> torch.Tensor:  # pragma: no cover - sanity guard
        pytest.fail("Projected gradient fallback should not be triggered when QP solver is requested.")

    monkeypatch.setattr(IMGradStrategy, "_solve_mgda_qp", fake_qp)
    monkeypatch.setattr(IMGradStrategy, "_mgda_weights_pgd", fail_pgd)
    strategy = IMGradStrategy(TASKS, solver="qp")
    update = strategy.compute_update_vector(conflicting_gradients)
    assert call_counter["qp"] == 1
    assert torch.isfinite(update).all()


def test_imgrad_qp_handles_float32_inputs(monkeypatch, conflicting_gradients: Dict[str, torch.Tensor]) -> None:
    dtype_records: List[np.dtype] = []

    class DummyMatrix:
        def __init__(self, data: object) -> None:
            array = np.asarray(data)
            dtype_records.append(array.dtype)
            self.data = array

        def __len__(self) -> int:
            return int(self.data.size)

    class DummySolvers:
        def __init__(self) -> None:
            self.options: Dict[str, object] = {}

        def qp(self, P, q, G, h, A, b):  # pragma: no cover - simple stub
            size = q.data.size if isinstance(q, DummyMatrix) else len(q)
            solution = np.full((size, 1), 1.0 / max(size, 1), dtype=np.float64)
            return {"status": "optimal", "x": solution}

    monkeypatch.setitem(_SOLVERS_AVAILABLE, "cvxopt", True)
    monkeypatch.setattr("coral_mtl.engine.gradient_strategies.matrix", lambda data: DummyMatrix(data))
    monkeypatch.setattr("coral_mtl.engine.gradient_strategies.solvers", DummySolvers())

    strategy = IMGradStrategy(TASKS, solver="qp")
    update = strategy.compute_update_vector(conflicting_gradients)

    assert dtype_records, "Expected cvxopt.matrix to be invoked at least once"
    assert all(dtype == np.float64 for dtype in dtype_records)
    assert torch.isfinite(update).all()


def test_nash_iterative_fallback_produces_finite_update(conflicting_gradients: Dict[str, torch.Tensor]) -> None:
    strategy = NashMTLStrategy(TASKS, solver="iterative", optim_niter=15)
    update = strategy.compute_update_vector(conflicting_gradients)
    assert torch.isfinite(update).all()
    assert update.shape == next(iter(conflicting_gradients.values())).shape


def test_nash_auto_solver_respects_availability(monkeypatch, conflicting_gradients: Dict[str, torch.Tensor]) -> None:
    monkeypatch.setitem(_SOLVERS_AVAILABLE, "cvxpy", False)
    call_counter = {"iterative": 0}

    def fake_iterative(self, gradients: torch.Tensor) -> torch.Tensor:  # pragma: no cover - monkeypatched
        call_counter["iterative"] += 1
        task_count = gradients.shape[0]
        return torch.full((task_count,), 1.0 / task_count, device=gradients.device, dtype=gradients.dtype)

    monkeypatch.setattr(NashMTLStrategy, "_solve_nash_iterative", fake_iterative)
    strategy = NashMTLStrategy(TASKS, solver="auto", optim_niter=10)
    update = strategy.compute_update_vector(conflicting_gradients)
    assert call_counter["iterative"] == 1
    assert not strategy.use_ccp
    assert torch.isfinite(update).all()


def test_nash_max_norm_clamps_update(conflicting_gradients: Dict[str, torch.Tensor]) -> None:
    scaled_gradients = {task: grad * 50 for task, grad in conflicting_gradients.items()}
    strategy = NashMTLStrategy(TASKS, solver="iterative", optim_niter=10, max_norm=0.5)
    update = strategy.compute_update_vector(scaled_gradients)
    assert torch.isfinite(update).all()
    assert update.norm().item() <= 0.5 + 1e-3


def test_nash_update_frequency_caches_solver_calls(monkeypatch, conflicting_gradients: Dict[str, torch.Tensor]) -> None:
    call_counter = {"iterative": 0}

    def fake_iterative(self, gradients: torch.Tensor) -> torch.Tensor:  # pragma: no cover - monkeypatched
        call_counter["iterative"] += 1
        task_count = gradients.shape[0]
        return torch.full((task_count,), 1.0 / task_count, device=gradients.device, dtype=gradients.dtype)

    monkeypatch.setattr(NashMTLStrategy, "_solve_nash_iterative", fake_iterative)
    strategy = NashMTLStrategy(TASKS, solver="iterative", update_frequency=2)
    strategy.compute_update_vector(conflicting_gradients)
    assert call_counter["iterative"] == 1
    strategy.compute_update_vector(conflicting_gradients)
    assert call_counter["iterative"] == 1
    strategy.compute_update_vector(conflicting_gradients)
    assert call_counter["iterative"] == 2


def test_nash_iterative_scale_invariance(conflicting_gradients: Dict[str, torch.Tensor]) -> None:
    strategy = NashMTLStrategy(TASKS, solver="iterative", optim_niter=25)
    base_update = strategy.compute_update_vector(conflicting_gradients)

    scaled_inputs = {task: grad * 3.5 for task, grad in conflicting_gradients.items()}
    scaled_strategy = NashMTLStrategy(TASKS, solver="iterative", optim_niter=25)
    scaled_update = scaled_strategy.compute_update_vector(scaled_inputs)

    base_norm = base_update.norm()
    scaled_norm = scaled_update.norm()
    assert base_norm > 0
    assert scaled_norm > 0

    cosine_similarity = F.cosine_similarity(base_update.unsqueeze(0), scaled_update.unsqueeze(0)).item()
    assert cosine_similarity > 0.95
    ratio = (scaled_norm / base_norm).item()
    assert ratio == pytest.approx(3.5, rel=5e-2)


@pytest.mark.optdeps
@pytest.mark.skipif(not _SOLVERS_AVAILABLE["cvxpy"], reason="cvxpy optional dependency not installed")
def test_nash_ccp_solver_invoked_when_available(monkeypatch, conflicting_gradients: Dict[str, torch.Tensor]) -> None:
    call_counter = {"ccp": 0}

    def fake_ccp(self, gradients: torch.Tensor) -> torch.Tensor:  # pragma: no cover - monkeypatched
        call_counter["ccp"] += 1
        task_count = gradients.shape[0]
        return torch.full((task_count,), 1.0 / task_count, device=gradients.device, dtype=gradients.dtype)

    def fail_iterative(self, gradients: torch.Tensor) -> torch.Tensor:  # pragma: no cover - sanity guard
        pytest.fail("Iterative fallback should not be triggered when CCP solver is requested.")

    monkeypatch.setattr(NashMTLStrategy, "_solve_nash_ccp", fake_ccp)
    monkeypatch.setattr(NashMTLStrategy, "_solve_nash_iterative", fail_iterative)
    strategy = NashMTLStrategy(TASKS, solver="ccp", optim_niter=5)
    update = strategy.compute_update_vector(conflicting_gradients)
    assert call_counter["ccp"] == 1
    assert torch.isfinite(update).all()


@pytest.mark.parametrize(
    ("strategy_cls", "kwargs"),
    [
        (IMGradStrategy, {"solver": "pgd", "mgda_pg_steps": 15}),
        (NashMTLStrategy, {"solver": "iterative", "optim_niter": 20}),
    ],
)
def test_gradient_strategies_reduce_combined_loss(
    strategy_cls,
    kwargs,
    tiny_mtl_setup,
    device: torch.device,
) -> None:
    torch.manual_seed(0)
    model, tasks, num_classes = tiny_mtl_setup
    strategy = strategy_cls(tasks, **kwargs)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    inputs, targets = _random_batch(tasks, num_classes, device)

    def total_unweighted_loss() -> float:
        predictions = model(inputs)
        losses = {
            task: criterion(predictions[task], targets[task])
            for task in tasks
        }
        return sum(loss.item() for loss in losses.values())

    before = total_unweighted_loss()

    predictions = model(inputs)
    per_task_losses = {
        task: criterion(predictions[task], targets[task])
        for task in tasks
    }
    params = [param for param in model.parameters() if param.requires_grad]
    per_task_gradients = _collect_per_task_gradients(model, per_task_losses, params, strategy.tasks)

    update_vector = strategy.compute_update_vector(per_task_gradients)
    optimizer.zero_grad()
    offset = 0
    for param in params:
        count = param.numel()
        param.grad = update_vector[offset:offset + count].view_as(param).clone()
        offset += count
    optimizer.step()

    after = total_unweighted_loss()
    assert after <= before + 1e-3


def test_gradient_strategies_manual_update_interface(conflicting_gradients: Dict[str, torch.Tensor]) -> None:
    strategy = NashMTLStrategy(TASKS, solver="iterative")
    assert hasattr(strategy, "requires_manual_backward_update")
    assert hasattr(strategy, "manual_backward_update")
    assert strategy.requires_manual_backward_update() is False
    strategy.manual_backward_update(nn.Linear(4, 2))  # should be a no-op
