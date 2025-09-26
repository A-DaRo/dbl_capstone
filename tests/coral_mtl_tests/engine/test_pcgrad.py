from __future__ import annotations

import torch

from coral_mtl.engine.pcgrad import PCGrad


def test_pcgrad_resolves_conflicting_gradients(device):  # noqa: D401
    param = torch.nn.Parameter(torch.tensor([1.0, 0.0], device=device))
    optimizer = torch.optim.SGD([param], lr=1.0)
    pcgrad = PCGrad(optimizer)

    vec_pos = torch.tensor([1.0, 0.0], device=device)
    vec_neg = torch.tensor([-1.0, 0.0], device=device)
    loss_a = torch.dot(param, vec_pos)
    loss_b = torch.dot(param, vec_neg)

    grads = [
        torch.autograd.grad(loss_a, [param], retain_graph=True),
        torch.autograd.grad(loss_b, [param], retain_graph=True)
    ]

    projected = pcgrad._project([list(g) for g in grads])
    combined = sum(g[0] for g in projected)
    dot_a = torch.dot(combined, grads[0][0])
    dot_b = torch.dot(combined, grads[1][0])
    assert dot_a >= -1e-6
    assert dot_b >= -1e-6


def test_pcgrad_projection(device):
    """Direct projection should remove conflicting component between gradients."""
    param = torch.nn.Parameter(torch.zeros(2, device=device))
    optimizer = torch.optim.SGD([param], lr=1.0)
    pcgrad = PCGrad(optimizer)

    g1 = torch.tensor([1.0, 0.5], device=device)
    g2 = torch.tensor([-1.0, 0.5], device=device)
    assert torch.dot(g1, g2) < 0

    projected = pcgrad._project([[g1.clone()], [g2.clone()]])
    g1_projected = projected[0][0]

    assert torch.dot(g1_projected, g2) >= -1e-6


def test_pcgrad_step_averages_projected_gradients(device):
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.zeros(2, device=device))
    optimizer = torch.optim.SGD([param], lr=1.0)
    pcgrad = PCGrad(optimizer)

    grad_task_a = torch.tensor([1.0, 0.0], device=device)
    grad_task_b = torch.tensor([-1.0, 0.0], device=device)
    task_grads = [[grad_task_a.clone()], [grad_task_b.clone()]]

    torch.manual_seed(0)
    projected = pcgrad._project([[grad_task_a.clone()], [grad_task_b.clone()]])
    expected_grad = torch.stack([proj[0] for proj in projected]).mean(dim=0)

    torch.manual_seed(0)
    pcgrad.zero_grad()
    pcgrad.step(task_grads, [param])

    assert torch.allclose(param.grad, expected_grad, atol=1e-6)
    assert torch.allclose(param.data, -expected_grad, atol=1e-6)