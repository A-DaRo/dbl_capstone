"""Gradient-based multi-task update strategies.

This module introduces strategies that combine per-task gradients into a
single update direction rather than forming a weighted scalar loss prior
to backpropagation. These are complementary to :class:`WeightingStrategy`.

Two execution modes are provided:

* **High-fidelity solvers** powered by optional convex optimisation
  libraries (`cvxopt` for MGDA within IMGrad and `cvxpy` for the CCP solver
  used by Nash-MTL). When these dependencies are installed they are used
  automatically for numerically robust behaviour.
* **Portable fallbacks** implemented entirely with PyTorch that approximate
  the reference algorithms, allowing the project to run in environments
  without the extra solver packages.

Solver availability is detected at import time and logged, while callers may
still override the behaviour explicitly via configuration.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

Tensor = torch.Tensor

logger = logging.getLogger(__name__)

_SOLVERS_AVAILABLE = {
    'cvxpy': False,
    'cvxopt': False,
}

try:  # pragma: no cover - optional dependency probing
    import cvxpy  # type: ignore

    _SOLVERS_AVAILABLE['cvxpy'] = True
except ImportError:  # pragma: no cover - log information only
    logger.info("`cvxpy` not found. NashMTLStrategy will use an iterative fallback solver.")

try:  # pragma: no cover - optional dependency probing
    import cvxopt  # type: ignore
    from cvxopt import matrix, solvers  # type: ignore

    solvers.options['show_progress'] = False
    _SOLVERS_AVAILABLE['cvxopt'] = True
except ImportError:  # pragma: no cover - log information only
    logger.info("`cvxopt` not found. IMGradStrategy will use a PGD-based fallback for the MGDA step.")


class GradientUpdateStrategy(nn.Module, ABC):
    """Abstract base for per-task gradient combination strategies.

    Workflow expectation:
      1. Training loop computes unweighted task losses L_t.
      2. For each task, backward retain_graph to obtain gradient g_t over shared params.
      3. Flatten gradients into vectors -> per_task_gradients dict.
      4. Strategy computes final update vector d (same flattened shape) via compute_update_vector.
      5. Trainer maps d back onto parameter .grad buffers then calls optimizer.step().
    """
    def __init__(self, tasks: List[str]):
        super().__init__()
        self.tasks = list(tasks)
        self._last_task_order: List[str] = []
        self._last_weights: Optional[torch.Tensor] = None
        self._last_metrics: Dict[str, Any] = {}

    @abstractmethod
    def compute_update_vector(self, per_task_gradients: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError

    def post_step(self):  # optional hook
        pass

    def _record_weights(self, task_names: List[str], weights: Tensor) -> None:
        self._last_task_order = list(task_names)
        self._last_weights = weights.detach().cpu()
        self._last_metrics['task_weights'] = {
            task: float(self._last_weights[idx])
            for idx, task in enumerate(self._last_task_order)
        }

    def get_last_weights(self) -> Optional[Dict[str, float]]:
        if self._last_weights is None or not self._last_task_order:
            return None
        return {
            task: float(self._last_weights[idx])
            for idx, task in enumerate(self._last_task_order)
        }

    def _record_metric(self, name: str, value: Any) -> None:
        self._last_metrics[name] = value

    def get_last_diagnostics(self) -> Dict[str, Any]:
        diagnostics = dict(self._last_metrics)
        if 'task_weights' not in diagnostics and self._last_weights is not None:
            diagnostics['task_weights'] = {
                task: float(self._last_weights[idx])
                for idx, task in enumerate(self._last_task_order)
            }
        return diagnostics


# ---- Utility helpers ----

def _stack_grads(per_task_gradients: Dict[str, Tensor]) -> Tensor:
    if not per_task_gradients:
        raise ValueError("No per-task gradients provided")
    return torch.stack([per_task_gradients[t] for t in per_task_gradients.keys()], dim=0)  # (T, D)


def _safe_norm(x: Tensor, eps: float = 1e-12) -> Tensor:
    return torch.sqrt(torch.clamp((x * x).sum(), min=eps))


class IMGradStrategy(GradientUpdateStrategy):
    """Imbalance-Sensitive Gradient Descent (IMGrad).

    Implements the minimum-norm MGDA step precisely when ``cvxopt`` is available
    and falls back to a projected-gradient approximation otherwise. The final
    update direction follows the blending heuristic proposed in the paper.
    """

    def __init__(
        self,
        tasks: List[str],
        mgda_pg_steps: int = 25,
        mgda_lr: float = 0.25,
        solver: str = 'auto',
    ):
        super().__init__(tasks)
        self.mgda_pg_steps = mgda_pg_steps
        self.mgda_lr = mgda_lr

        solver_lower = solver.lower()
        if solver_lower == 'auto':
            self.use_qp = _SOLVERS_AVAILABLE['cvxopt']
        elif solver_lower == 'qp':
            if not _SOLVERS_AVAILABLE['cvxopt']:
                raise ImportError("`cvxopt` is required for solver='qp' but is not installed.")
            self.use_qp = True
        elif solver_lower == 'pgd':
            self.use_qp = False
        else:
            raise ValueError("solver must be one of {'auto', 'qp', 'pgd'}")

        logger.info(
            "Initialized IMGradStrategy with solver: %s",
            'QP (cvxopt)' if self.use_qp else 'Projected Gradient Fallback',
        )

    def _mgda_weights_pgd(self, G: Tensor) -> Tensor:
        # G: (T, D)
        T = G.shape[0]
        w = torch.full((T,), 1.0 / max(T, 1), device=G.device, dtype=G.dtype)
        if T <= 1:
            return w
        for _ in range(self.mgda_pg_steps):
            GGt = G @ G.t()  # (T,T)
            grad = 2 * (GGt @ w)
            w = w - self.mgda_lr * grad
            w = torch.clamp(w, min=0)
            sum_w = w.sum()
            if sum_w <= 0:
                w.fill_(1.0 / T)
            else:
                w = w / sum_w
        return w

    def _solve_mgda_qp(self, G: Tensor) -> Tensor:
        if not _SOLVERS_AVAILABLE['cvxopt']:
            raise RuntimeError("QP solver requested but `cvxopt` is unavailable.")
        GTG = (G @ G.t()).detach().cpu().numpy()
        T = GTG.shape[0]
        if T == 1:
            return torch.ones(1, device=G.device, dtype=G.dtype)
        P = matrix(GTG)
        q = matrix(torch.zeros(T, dtype=torch.double).numpy())
        G_constr = matrix(-torch.eye(T, dtype=torch.double).numpy())
        h_constr = matrix(torch.zeros(T, dtype=torch.double).numpy())
        A_constr = matrix(torch.ones(1, T, dtype=torch.double).numpy())
        b_constr = matrix(torch.ones(1, dtype=torch.double).numpy())
        try:
            solution = solvers.qp(P, q, G_constr, h_constr, A_constr, b_constr)
        except Exception as exc:  # pragma: no cover - depends on solver internals
            logger.warning("IMGradStrategy QP solver failed (%s); falling back to PGD approximation.", exc)
            return self._mgda_weights_pgd(G)
        status = solution.get('status', '')
        if status != 'optimal':
            logger.warning("IMGradStrategy QP solver status '%s'; using PGD fallback.", status)
            return self._mgda_weights_pgd(G)
        weights_np = np.array(solution['x'], dtype=np.float64).reshape(-1)
        weights = torch.from_numpy(weights_np).to(device=G.device, dtype=G.dtype)
        weights = torch.clamp(weights, min=0)
        sum_w = weights.sum()
        if sum_w <= 0:
            return self._mgda_weights_pgd(G)
        return weights / sum_w

    def compute_update_vector(self, per_task_gradients: Dict[str, Tensor]) -> Tensor:
        self._last_metrics = {}
        task_names = list(per_task_gradients.keys())
        G = torch.stack([per_task_gradients[t] for t in task_names], dim=0)  # (T,D)
        if G.shape[0] == 1:
            self._record_weights(task_names, torch.ones((1,), device=G.device, dtype=G.dtype))
            return G[0]
        if self.use_qp:
            weights = self._solve_mgda_qp(G)
        else:
            weights = self._mgda_weights_pgd(G)
        self._record_weights(task_names, weights)
        gm = (weights.unsqueeze(1) * G).sum(dim=0)
        g0 = G.mean(dim=0)
        gm_norm = _safe_norm(gm)
        if gm_norm <= 0:
            self._record_metric('imgrad_cos_theta', 0.0)
            return g0
        cos_theta = torch.clamp(torch.dot(g0, gm) / (_safe_norm(g0) * gm_norm), -1.0, 1.0)
        self._record_metric('imgrad_cos_theta', float(cos_theta.detach().item()))
        alpha = 0.5 * (1 - cos_theta)
        d = (1 - alpha) * g0 + alpha * gm
        self._record_metric('gradient_update_norm', float(torch.norm(d).detach().item()))
        return d


class NashMTLStrategy(GradientUpdateStrategy):
    """Proportionally fair multi-task optimisation (Nash-MTL).

    Uses the Concave-Convex Procedure (CCP) with ``cvxpy`` when available and
    falls back to an iterative reweighted least-squares scheme otherwise. The
    strategy caches task weights and recomputes them at a configurable cadence
    for efficiency.
    """

    def __init__(
        self,
        tasks: List[str],
        update_frequency: int = 1,
        optim_niter: int = 20,
        solver: str = 'auto',
        max_norm: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__(tasks)
        self.update_frequency = max(1, update_frequency)
        self.optim_niter = optim_niter
        self.max_norm = max_norm
        self.eps = eps
        self.n_tasks = len(tasks)

        solver_lower = solver.lower()
        if solver_lower == 'auto':
            self.use_ccp = _SOLVERS_AVAILABLE['cvxpy']
        elif solver_lower == 'ccp':
            if not _SOLVERS_AVAILABLE['cvxpy']:
                raise ImportError("`cvxpy` is required for solver='ccp' but is not installed.")
            self.use_ccp = True
        elif solver_lower in {'iterative', 'fallback'}:
            self.use_ccp = False
        else:
            raise ValueError("solver must be one of {'auto', 'ccp', 'iterative'}")

        initial_weights = torch.full((self.n_tasks,), 1.0 / max(1, self.n_tasks))
        self.register_buffer('_cached_weights', initial_weights)
        self._step = 0

        self._cvxpy_problem = None
        self._cvxpy_vars: Dict[str, Any] = {}

        logger.info(
            "Initialized NashMTLStrategy with solver: %s",
            'CCP (cvxpy)' if self.use_ccp else 'Iterative Fallback',
        )

    def _ensure_ccp_problem(self, T: int) -> None:
        if self._cvxpy_problem is not None:
            return
        if not _SOLVERS_AVAILABLE['cvxpy']:
            raise RuntimeError("CCP solver requested but `cvxpy` is unavailable.")
        import cvxpy as cp  # type: ignore

        alpha = cp.Variable(T, nonneg=True)
        prev_alpha = cp.Parameter(T, nonneg=True)
        linear_grad = cp.Parameter(T)
        G_param = cp.Parameter((T, T))

        objective = cp.Minimize(cp.sum(G_param @ alpha) + linear_grad @ (alpha - prev_alpha))
        constraints = [
            alpha >= self.eps,
            G_param @ alpha >= self.eps,
            cp.sum(alpha) == 1,
        ]
        problem = cp.Problem(objective, constraints)

        self._cvxpy_problem = problem
        self._cvxpy_vars = {
            'alpha': alpha,
            'prev_alpha': prev_alpha,
            'linear_grad': linear_grad,
            'G_param': G_param,
        }

    def _solve_nash_ccp(self, G: Tensor) -> Tensor:
        if not _SOLVERS_AVAILABLE['cvxpy']:
            raise RuntimeError("CCP solver requested but `cvxpy` is unavailable.")
        import cvxpy as cp  # type: ignore

        T = G.shape[0]
        self._ensure_ccp_problem(T)
        GTG = (G @ G.t()).detach().cpu().numpy()
        norm_factor = np.linalg.norm(GTG)
        if norm_factor <= 0:
            norm_factor = 1.0
        GTG_norm = GTG / norm_factor

        prev_alpha_np = self._cached_weights.detach().cpu().numpy()
        alpha_t = prev_alpha_np.copy()

        vars = self._cvxpy_vars
        vars['G_param'].value = GTG_norm

        for _ in range(self.optim_niter):
            vars['prev_alpha'].value = np.clip(alpha_t, self.eps, None)
            G_prev_alpha = GTG_norm @ vars['prev_alpha'].value
            G_prev_alpha = np.clip(G_prev_alpha, self.eps, None)
            linear_grad = (1.0 / vars['prev_alpha'].value) + GTG_norm.T @ (1.0 / G_prev_alpha)
            vars['linear_grad'].value = linear_grad

            try:
                self._cvxpy_problem.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except cp.error.SolverError as exc:  # pragma: no cover - depends on solver
                logger.warning("NashMTLStrategy CCP solver failed (%s); reusing previous weights.", exc)
                return torch.from_numpy(prev_alpha_np).to(G.device, dtype=G.dtype)

            if vars['alpha'].value is None:
                logger.warning("NashMTLStrategy CCP returned no solution; reusing previous weights.")
                return torch.from_numpy(prev_alpha_np).to(G.device, dtype=G.dtype)

            alpha_t = np.clip(vars['alpha'].value, self.eps, None)
            residual = np.linalg.norm(GTG_norm @ alpha_t - 1.0 / (alpha_t + 1e-10))
            if residual < 1e-3:
                break

        alpha_tensor = torch.from_numpy(alpha_t).to(G.device, dtype=G.dtype)
        return alpha_tensor

    def _solve_nash_iterative(self, G: Tensor) -> Tensor:
        a = self._cached_weights.clone().to(G.device, dtype=G.dtype)
        if self.n_tasks <= 1:
            return a
        for _ in range(self.optim_niter):
            Ga = (a.unsqueeze(1) * G).sum(dim=0)
            dot_terms = G @ Ga
            grad = dot_terms - 1.0 / torch.clamp(a, min=self.eps)
            step = 0.1 / (grad.abs().mean() + 1e-4)
            a = a - step * grad
            a = torch.clamp(a, min=self.eps)
            a = a / a.sum()
        return a

    def compute_update_vector(self, per_task_gradients: Dict[str, Tensor]) -> Tensor:
        self._last_metrics = {}
        if not per_task_gradients:
            raise ValueError("No gradients provided to NashMTLStrategy")
        task_names = list(per_task_gradients.keys())
        G = torch.stack([per_task_gradients[t] for t in task_names], dim=0)
        if G.shape[0] == 0:
            return torch.zeros_like(next(iter(per_task_gradients.values())))
        if G.shape[0] == 1:
            self._record_weights(task_names, torch.ones((1,), device=G.device, dtype=G.dtype))
            return G[0]

        self._step += 1
        recompute = (self._step - 1) % self.update_frequency == 0
        if recompute:
            if self.use_ccp:
                weights = self._solve_nash_ccp(G)
            else:
                weights = self._solve_nash_iterative(G)
            weights = torch.clamp(weights, min=self.eps)
            sum_w = weights.sum()
            if sum_w <= 0:
                weights = torch.full((self.n_tasks,), 1.0 / max(1, self.n_tasks), device=G.device, dtype=G.dtype)
            else:
                weights = weights / sum_w
            self._cached_weights = weights.detach().cpu()

        weights = self._cached_weights.to(G.device, dtype=G.dtype)
        self._record_weights(task_names, weights)
        update = (weights.unsqueeze(1) * G).sum(dim=0)
        if self.max_norm > 0:
            norm = torch.norm(update)
            if norm > self.max_norm:
                update = update * (self.max_norm / (norm + 1e-6))
        self._record_metric('gradient_update_norm', float(torch.norm(update).detach().item()))
        return update

    def post_step(self):
        pass


__all__ = [
    'GradientUpdateStrategy',
    'IMGradStrategy',
    'NashMTLStrategy'
]
