import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_polynomial_decay_schedule_with_warmup
from typing import Tuple, Dict, Any

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

    This function separates model parameters into two groups: one with weight decay
    and one without (for biases and normalization layer parameters), which is a
    common best practice for training Transformer-based models.

    Args:
        model (nn.Module): The model to be optimized.
        learning_rate (float): The initial peak learning rate.
        weight_decay (float): The weight decay value for non-bias/norm parameters.
        adam_betas (Tuple[float, float]): The beta values for the AdamW optimizer.
        num_training_steps (int): The total number of training steps.
        num_warmup_steps (int): The number of steps for the linear warmup phase.
        power (float): The power for the polynomial decay. 1.0 is linear decay.

    Returns:
        Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
            - The configured AdamW optimizer.
            - The configured learning rate scheduler.
    """
    # --- 1. Separate parameters for weight decay ---
    # We don't want to apply weight decay to bias terms or LayerNorm/BatchNorm weights.
    decay_parameters = []
    no_decay_parameters = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check for parameters to exclude from weight decay
        if param.dim() == 1 or name.endswith(".bias"):
            no_decay_parameters.append(param)
        else:
            decay_parameters.append(param)
            
    # --- 2. Create optimizer parameter groups ---
    optimizer_grouped_parameters = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]

    # --- 3. Instantiate the AdamW optimizer ---
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=adam_betas
    )

    # --- 4. Instantiate the learning rate scheduler ---
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        power=power,
        lr_end=1e-7 # The learning rate will decay towards this value
    )

    return optimizer, scheduler


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("--- Running Sanity Check for Optimizer and Scheduler ---")

    # 1. Create a dummy model to test parameter grouping
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.ln = nn.LayerNorm(16)
            self.linear = nn.Linear(16, 10)
            self.bn = nn.BatchNorm1d(10)
    
    dummy_model = DummyModel()
    
    # 2. Define training hyperparameters
    total_epochs = 50
    steps_per_epoch = 200
    total_training_steps = total_epochs * steps_per_epoch
    warmup_steps = 1500
    lr = 6e-5

    # 3. Create the optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=dummy_model,
        learning_rate=lr,
        num_training_steps=total_training_steps,
        num_warmup_steps=warmup_steps
    )

    # 4. Verify the optimizer parameter groups
    print("\n--- Optimizer Parameter Groups ---")
    param_groups = optimizer.param_groups
    print(f"Number of parameter groups: {len(param_groups)}")
    print(f"Group 1 (with weight decay): {param_groups[0]['weight_decay']}")
    print(f"Group 2 (without weight decay): {param_groups[1]['weight_decay']}")
    assert param_groups[0]['weight_decay'] > 0
    assert param_groups[1]['weight_decay'] == 0
    print("Parameter grouping for weight decay is correct.")

    # 5. Simulate training and record learning rate at each step
    print("\nSimulating training to visualize LR schedule...")
    learning_rates = []
    for step in range(total_training_steps):
        # In a real training loop, you would do:
        # loss.backward()
        # optimizer.step()
        # scheduler.step()
        # optimizer.zero_grad()
        
        # For simulation, we just update the scheduler
        learning_rates.append(scheduler.get_last_lr()[0])
        optimizer.step() # Dummy step to avoid warning
        scheduler.step()

    # 6. Plot the learning rate schedule
    plt.figure(figsize=(12, 6))
    plt.plot(learning_rates)
    plt.axvline(x=warmup_steps, color='r', linestyle='--', label='End of Warmup')
    plt.title("Polynomial Decay LR Schedule with Linear Warmup")
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.legend()
    
    print(f"\nPlotting learning rate schedule for {total_training_steps} steps.")
    print("The plot shows the linear warmup phase followed by polynomial decay.")
    plt.show()
    
    print("\nSanity check passed! Optimizer and scheduler created successfully.")