"""Loss weighting strategy abstractions for multi-task learning.

This module introduces a strategy pattern layer between raw per-task loss
computation and their aggregation into a final optimized scalar. It enables
rapid experimentation with alternative balancing mechanisms (e.g., homoscedastic
uncertainty, GradNorm, DWA, PCGrad-style post-hoc adjustments) without
modifying the core `CoralMTLLoss` implementation.

 - Provide abstract `WeightingStrategy` contract.
 - Migrate existing uncertainty weighting logic into `UncertaintyWeightingStrategy`.
 - Ensure backward compatibility: absence of explicit strategy config will
   default to uncertainty weighting with identical semantics to the pre-refactor
   behavior (so existing configs remain valid).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Iterable

import torch
import torch.nn as nn

from .gradient_strategies import IMGradStrategy, NashMTLStrategy


class WeightingStrategy(nn.Module, ABC):
    """Abstract base class for multi-task loss weighting strategies.

    Contract:
      forward(unweighted_losses) -> Dict[str, torch.Tensor]
        Must return a dictionary containing at minimum a 'total_loss' tensor.
        Additional keys are free-form but should be scalar tensors used for
        logging/analysis (e.g., per-task weighted contributions, learned params).
    """

    def __init__(self, tasks: List[str]):
        super().__init__()
        self.tasks = list(tasks)
        self._latest_losses: Dict[str, torch.Tensor] = {}
        self._latest_diagnostics: Dict[str, Any] = {}

    @abstractmethod
    def forward(self, unweighted_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # noqa: D401
        raise NotImplementedError

    # --- Optional hooks for strategies that need additional signals ---
    def cache_unweighted_losses(self, losses: Dict[str, torch.Tensor]) -> None:
        """Store references to the most recent unweighted losses (no detach)."""
        self._latest_losses = losses

    def update_epoch_losses(self, epoch_loss_means: Dict[str, float], *, epoch: int) -> None:
        """Hook receiving averaged training losses once per epoch (default noop)."""

    def requires_manual_backward_update(self) -> bool:
        return False

    def manual_backward_update(self, model: nn.Module) -> None:
        """Called before optimizer.step when manual gradient-based updates are required."""

    # --- Diagnostics helpers ---
    def _reset_diagnostics(self) -> None:
        self._latest_diagnostics = {}

    def _record_task_weight(self, task: str, value: float) -> None:
        diag = self._latest_diagnostics.setdefault('task_weights', {})
        diag[task] = float(value)

    def _record_log_variance(self, task: str, value: float) -> None:
        diag = self._latest_diagnostics.setdefault('log_variances', {})
        diag[task] = float(value)

    def _record_aux_metric(self, name: str, value: Any) -> None:
        self._latest_diagnostics[name] = value

    def get_diagnostics(self) -> Dict[str, Any]:
        return dict(self._latest_diagnostics)

    def get_last_diagnostics(self) -> Dict[str, Any]:
        return self.get_diagnostics()


def _ensure_tensor_list(tensors: Iterable[Optional[torch.Tensor]]) -> List[torch.Tensor]:
    return [t for t in tensors if t is not None]


class UncertaintyWeightingStrategy(WeightingStrategy):
    """Task-wise homoscedastic uncertainty weighting (Kendall et al. 2018).

    Each task t owns a learnable log variance parameter log_var_t. The weighted
    contribution of task loss L_t is:

        exp(-log_var_t) * L_t + 0.5 * log_var_t

    This encourages tasks with higher predictive noise (captured by larger
    log_var_t) to receive smaller relative weights while regularising the
    parameter growth via the additive 0.5 * log_var_t term.
    """

    def __init__(
        self,
        tasks: List[str],
        clamp_range: Optional[float] = 10.0,
        learnable_tasks: Optional[Iterable[str]] = None,
    ):
        super().__init__(tasks=tasks)
        allowed = set(tasks)
        if learnable_tasks is None:
            learnable = allowed
        else:
            learnable = {task for task in learnable_tasks if task in allowed}
        self.learnable_tasks = learnable
        # ParameterDict keyed by task for easy logging & selective freezing
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros((), dtype=torch.float32)) for task in learnable
        })
        self.clamp_range = clamp_range

    def forward(self, unweighted_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # noqa: D401
        self.cache_unweighted_losses(unweighted_losses)
        self._reset_diagnostics()
        if not unweighted_losses:
            raise ValueError("UncertaintyWeightingStrategy received empty unweighted_losses dict")
        device = next(iter(unweighted_losses.values())).device
        total = torch.zeros((), device=device)
        out: Dict[str, torch.Tensor] = {}
        for task, loss in unweighted_losses.items():
            if task in self.log_vars:
                log_var = self.log_vars[task]
                if self.clamp_range is not None:
                    log_var_eff = torch.clamp(log_var, -self.clamp_range, self.clamp_range)
                else:
                    log_var_eff = log_var
                precision = torch.exp(-log_var_eff)
                weighted = precision * loss + 0.5 * log_var_eff
                out[f'log_var_{task}'] = log_var.detach()
                self._record_task_weight(task, float(precision.detach().item()))
                self._record_log_variance(task, float(log_var_eff.detach().item()))
            else:
                weighted = loss
                self._record_task_weight(task, 1.0)
            total = total + weighted
            # Logging tensors (detach only where we don't need grads upstream)
            out[f'weighted_{task}_loss'] = weighted
        out['total_loss'] = total
        return out


class DWAWeightingStrategy(WeightingStrategy):
    """Dynamic Weight Averaging (Liu et al., 2019)."""

    def __init__(self, tasks: List[str], temperature: float = 2.0, eps: float = 1e-8):
        super().__init__(tasks)
        self.temperature = temperature
        self.eps = eps
        self.loss_history: Dict[str, List[float]] = {task: [] for task in tasks}

    def forward(self, unweighted_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # noqa: D401
        self.cache_unweighted_losses(unweighted_losses)
        self._reset_diagnostics()
        device = next(iter(unweighted_losses.values())).device
        weights = self._compute_weights(device)
        total = torch.zeros((), device=device)
        out: Dict[str, torch.Tensor] = {}
        for task, loss in unweighted_losses.items():
            weight = weights[task]
            weighted = weight * loss
            total = total + weighted
            out[f'weighted_{task}_loss'] = weighted
            out[f'dwa_weight_{task}'] = weight.detach()
            self._record_task_weight(task, float(weight.detach().item()))
        out['total_loss'] = total
        return out

    def _compute_weights(self, device: torch.device) -> Dict[str, torch.Tensor]:
        all_weights: Dict[str, torch.Tensor] = {}
        if all(len(h) >= 2 for h in self.loss_history.values() if len(self.tasks) > 1):
            ratios = []
            for task in self.tasks:
                last_two = self.loss_history[task][-2:]
                ratio = last_two[1] / max(last_two[0], self.eps)
                ratios.append(ratio)
            ratios_tensor = torch.tensor(ratios, device=device, dtype=torch.float32)
            exps = torch.exp(ratios_tensor / self.temperature)
            weights_tensor = exps / exps.sum()
            for idx, task in enumerate(self.tasks):
                all_weights[task] = weights_tensor[idx]
        else:
            uniform = torch.full((len(self.tasks),), 1.0 / max(len(self.tasks), 1), device=device)
            for idx, task in enumerate(self.tasks):
                all_weights[task] = uniform[idx]
        return all_weights

    def update_epoch_losses(self, epoch_loss_means: Dict[str, float], *, epoch: int) -> None:
        for task in self.tasks:
            key = f'unweighted_{task}_loss'
            if key in epoch_loss_means:
                self.loss_history[task].append(epoch_loss_means[key])
                if len(self.loss_history[task]) > 2:
                    self.loss_history[task] = self.loss_history[task][-2:]


class GradNormWeightingStrategy(WeightingStrategy):
    """GradNorm (Chen et al., 2018) balancing using manual weight updates."""

    def __init__(self,
                 tasks: List[str],
                 alpha: float = 0.5,
                 lr: float = 0.025,
                 eps: float = 1e-8):
        super().__init__(tasks)
        self.alpha = alpha
        self.lr = lr
        self.eps = eps
        self.weights = nn.ParameterDict({
            task: nn.Parameter(torch.ones((), dtype=torch.float32), requires_grad=False)
            for task in tasks
        })
        self.initial_losses: Optional[Dict[str, float]] = None
        self.shared_parameters: Optional[List[torch.nn.Parameter]] = None

    def forward(self, unweighted_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # noqa: D401
        if not unweighted_losses:
            raise ValueError("GradNorm requires at least one unweighted loss")
        self.cache_unweighted_losses(unweighted_losses)
        self._reset_diagnostics()
        device = next(iter(unweighted_losses.values())).device
        total = torch.zeros((), device=device)
        out: Dict[str, torch.Tensor] = {}
        for task, loss in unweighted_losses.items():
            weight = self.weights[task]
            weighted = weight * loss
            total = total + weighted
            out[f'weighted_{task}_loss'] = weighted
            out[f'gradnorm_weight_{task}'] = weight.detach()
            self._record_task_weight(task, float(weight.detach().item()))
        out['total_loss'] = total
        return out

    def requires_manual_backward_update(self) -> bool:
        return True

    def register_shared_parameters(self, params: List[torch.nn.Parameter]) -> None:
        self.shared_parameters = [p for p in params if p.requires_grad]

    def _ensure_shared_parameters(self, model: nn.Module) -> None:
        if self.shared_parameters is None:
            self.register_shared_parameters([p for p in model.parameters()])

    def manual_backward_update(self, model: nn.Module) -> None:
        self._ensure_shared_parameters(model)
        if not self.shared_parameters:
            return
        losses = self._latest_losses
        if not losses:
            return
        with torch.no_grad():
            if self.initial_losses is None:
                self.initial_losses = {task: float(loss.detach().item()) for task, loss in losses.items()}
                return
        grad_norms = []
        loss_values = []
        processed_tasks: List[str] = []
        for task in self.tasks:
            loss = losses.get(task)
            if loss is None:
                continue
            weighted = (self.weights[task] * loss)
            grads = torch.autograd.grad(
                weighted,
                self.shared_parameters,
                retain_graph=True,
                allow_unused=True
            )
            grads = _ensure_tensor_list(grads)
            if not grads:
                grad_norms.append(torch.zeros((), device=loss.device))
            else:
                grad_norms.append(torch.stack([g.norm(2) for g in grads]).sum())
            loss_values.append(loss.detach())
            processed_tasks.append(task)

        if not grad_norms:
            return
        grad_norms_tensor = torch.stack(grad_norms)
        avg_grad_norm = grad_norms_tensor.mean()
        loss_tensor = torch.stack(loss_values)
        init_tensor = torch.tensor([self.initial_losses[task] for task in processed_tasks], device=loss_tensor.device)
        ratio = loss_tensor / (init_tensor + self.eps)
        mean_ratio = ratio.mean()
        target = avg_grad_norm * (ratio / (mean_ratio + self.eps)) ** self.alpha

        with torch.no_grad():
            for idx, task in enumerate(processed_tasks):
                grad_diff = grad_norms_tensor[idx] - target[idx]
                self.weights[task].add_(-self.lr * grad_diff)
                self.weights[task].clamp_(min=self.eps)
            # Normalize weights to keep sum = N_tasks
            weights_tensor = torch.stack([self.weights[task] for task in self.tasks])
            scale = len(self.tasks) / (weights_tensor.sum() + self.eps)
            for idx, task in enumerate(self.tasks):
                self.weights[task].mul_(scale)
def build_weighting_strategy(config: Dict, primary: List[str], auxiliary: List[str]) -> WeightingStrategy:
    """Factory helper for ExperimentFactory.

    Parameters
    ----------
    config : Dict
        `loss.weighting_strategy` config block (may be None / missing).
    primary : List[str]
        Primary task names.
    auxiliary : List[str]
        Auxiliary task names (still weighted independently in strategy scope).

    Returns
    -------
    WeightingStrategy
        Instantiated strategy (currently only Uncertainty supported).
    """
    if config is None:
        config = {}
    strategy_type = (config.get('type') or 'Uncertainty').lower()
    params = config.get('params', {})
    all_tasks = primary + auxiliary
    if strategy_type == 'uncertainty':
        learnable = params.get('learnable_tasks', primary)
        return UncertaintyWeightingStrategy(
            all_tasks,
            clamp_range=params.get('clamp_range', 10.0),
            learnable_tasks=learnable,
        )
    if strategy_type == 'dwa':
        return DWAWeightingStrategy(all_tasks, temperature=params.get('temperature', 2.0))
    if strategy_type == 'gradnorm':
        return GradNormWeightingStrategy(
            all_tasks,
            alpha=params.get('alpha', 0.5),
            lr=params.get('lr', 0.025)
        )
    if strategy_type == 'imgrad':
        return IMGradStrategy(all_tasks,
                              mgda_pg_steps=params.get('mgda_pg_steps', 25),
                              mgda_lr=params.get('mgda_lr', 0.25),
                              solver=params.get('solver', 'auto'))
    if strategy_type == 'nashmtl':
        return NashMTLStrategy(
            all_tasks,
            update_frequency=params.get('update_frequency', 25),
            optim_niter=params.get('optim_niter', params.get('iters', 20)),
            solver=params.get('solver', 'auto'),
            max_norm=params.get('max_norm', 0.0),
            eps=params.get('eps', 1e-8),
        )
    raise ValueError(f"Unknown weighting strategy type '{strategy_type}'.")
