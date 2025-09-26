"""Projected Conflicting Gradient (PCGrad) wrapper implementation."""
from __future__ import annotations

from typing import Iterable, List, Sequence

import torch


def _flatten_grad_list(grad_list: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.view(-1) for g in grad_list if g is not None])


class PCGrad:
    """Wraps an optimizer to apply PCGrad projections before the step."""

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    @staticmethod
    def _dot(grad_a: Sequence[torch.Tensor], grad_b: Sequence[torch.Tensor]) -> torch.Tensor:
        dot = torch.zeros((), device=grad_a[0].device if grad_a and grad_a[0] is not None else torch.device('cpu'))
        for ga, gb in zip(grad_a, grad_b):
            if ga is None or gb is None:
                continue
            dot = dot + torch.sum(ga * gb)
        return dot

    @staticmethod
    def _norm_sq(grad: Sequence[torch.Tensor]) -> torch.Tensor:
        norm = torch.zeros((), device=grad[0].device if grad and grad[0] is not None else torch.device('cpu'))
        for g in grad:
            if g is None:
                continue
            norm = norm + g.pow(2).sum()
        return norm

    def _project(self, grads: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
        num_tasks = len(grads)
        if num_tasks <= 1:
            return grads
        order = torch.randperm(num_tasks)
        projected = [[g.clone() if g is not None else None for g in grad] for grad in grads]
        original = grads
        for idx in order:
            g_i = projected[idx]
            for jdx in order:
                if idx == jdx:
                    continue
                dot = self._dot(g_i, original[jdx])
                if dot < 0:
                    norm_sq = self._norm_sq(original[jdx])
                    if norm_sq > 0:
                        coeff = dot / norm_sq
                        g_i = [
                            gi - coeff * gj if gi is not None and gj is not None else gi
                            for gi, gj in zip(g_i, original[jdx])
                        ]
            projected[idx] = g_i
        return projected

    def step(self, task_grads: List[Iterable[torch.Tensor]], params: List[torch.nn.Parameter]) -> None:
        projected = self._project([list(g) for g in task_grads])
        final_grads: List[torch.Tensor | None] = []
        for grads_per_param in zip(*projected):
            grads_per_param = [g for g in grads_per_param if g is not None]
            if not grads_per_param:
                final_grads.append(None)
            else:
                stacked = torch.stack([g for g in grads_per_param])
                final_grads.append(stacked.mean(dim=0))
        for param, grad in zip(params, final_grads):
            if grad is None:
                param.grad = None
            else:
                param.grad = grad.clone()
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)