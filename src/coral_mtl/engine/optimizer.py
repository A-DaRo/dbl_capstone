import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_polynomial_decay_schedule_with_warmup
from typing import Tuple, Any

def create_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float = 6e-5,
    weight_decay: float = 0.01,
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    num_training_steps: int = 10000,
    num_warmup_steps: int = 1500,
    power: float = 1.0
) -> Tuple[optim.Optimizer, Any]:
    """
    Creates an AdamW optimizer and a polynomial decay learning rate scheduler with warmup.

    Separates model parameters into two groups: one with weight decay and one without
    (for biases and normalization layer parameters), a best practice for training
    Transformer-based models.
    """
    decay_parameters, no_decay_parameters = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() == 1 or name.endswith(".bias"):
            no_decay_parameters.append(param)
        else:
            decay_parameters.append(param)
            
    optimizer_grouped_parameters = [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, betas=adam_betas)

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        power=power,
        lr_end=1e-7
    )
    return optimizer, scheduler