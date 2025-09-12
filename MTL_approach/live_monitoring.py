# live_monitoring.py
# Dependencies: matplotlib, numpy, torch

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List

def plot_composite_losses(log_history: Dict[str, List[float]], save_path: str = None):
    """Plots total, primary, and auxiliary losses on a single axis."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    steps = range(len(log_history.get('total_loss', [])))
    ax.plot(steps, log_history.get('total_loss', []), label='Total Loss', color='b', linewidth=2)
    ax.plot(steps, log_history.get('primary_loss', []), label='Primary Loss', color='g', linestyle='--')
    ax.plot(steps, log_history.get('auxiliary_loss', []), label='Auxiliary Loss', color='r', linestyle=':')
    
    ax.set_title('Composite Loss Trends During Training', fontsize=16)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_yscale('log') # Log scale is often better for viewing loss
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_individual_task_losses(log_history: Dict[str, List[float]], save_path: str = None):
    """Creates a grid of plots for each individual task loss."""
    task_losses = {k: v for k, v in log_history.items() if '_loss' in k and k not in ['total_loss', 'primary_loss', 'auxiliary_loss', 'consistency_loss']}
    num_tasks = len(task_losses)
    if num_tasks == 0:
        print("No individual task losses found in log history.")
        return
        
    fig, axes = plt.subplots(num_tasks, 1, figsize=(10, 4 * num_tasks), sharex=True)
    if num_tasks == 1:
        axes = [axes] # Make it iterable
        
    steps = range(len(list(task_losses.values())[0]))
    
    for ax, (name, values) in zip(axes, task_losses.items()):
        ax.plot(steps, values, label=name)
        ax.set_title(f'Loss Trend for: {name}')
        ax.set_ylabel('Loss')
        ax.legend()

    axes[-1].set_xlabel('Training Steps')
    plt.suptitle('Individual Task Loss Diagnostics', fontsize=18, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_uncertainty_weights(log_history: Dict[str, List[float]], save_path: str = None):
    """Plots the learned uncertainty weights (sigma^2) for primary tasks."""
    log_var_genus = torch.tensor(log_history.get('log_var_genus', []))
    log_var_health = torch.tensor(log_history.get('log_var_health', []))
    
    # Convert log variance to sigma^2 for more intuitive plotting
    sigma_sq_genus = torch.exp(log_var_genus).numpy()
    sigma_sq_health = torch.exp(log_var_health).numpy()
    
    steps = range(len(sigma_sq_genus))
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(steps, sigma_sq_genus, label='σ² Genus (Uncertainty)', color='purple')
    ax.plot(steps, sigma_sq_health, label='σ² Health (Uncertainty)', color='orange')
    
    ax.set_title('Learned Task Uncertainty (σ²)', fontsize=16)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('σ² Value', fontsize=12)
    ax.legend(fontsize=10)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_learning_rate(log_history: Dict[str, List[float]], warmup_steps: int, save_path: str = None):
    """Plots the learning rate schedule to verify warmup and decay."""
    lr_values = log_history.get('lr', [])
    steps = range(len(lr_values))
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(steps, lr_values, label='Learning Rate', color='teal')
    ax.axvline(x=warmup_steps, color='r', linestyle='--', label='End of Warmup')
    
    ax.set_title('Learning Rate Schedule', fontsize=16)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.legend(fontsize=10)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    print("--- Running Demo of Live Monitoring Plots ---")
    
    # --- Create Dummy Log Data (simulating a 1000-step training run) ---
    num_steps = 1000
    warmup_steps = 150
    
    mock_log_history = {
        'total_loss': np.log(np.linspace(10, 1.5, num_steps)) + np.random.randn(num_steps) * 0.1,
        'primary_loss': np.log(np.linspace(8, 1, num_steps)) + np.random.randn(num_steps) * 0.1,
        'auxiliary_loss': np.log(np.linspace(5, 1, num_steps)) + np.random.randn(num_steps) * 0.1,
        'genus_loss': np.log(np.linspace(4.5, 0.6, num_steps)) + np.random.randn(num_steps) * 0.05,
        'health_loss': np.log(np.linspace(3.5, 0.4, num_steps)) + np.random.randn(num_steps) * 0.05,
        'fish_loss': np.log(np.linspace(2, 0.5, num_steps)) + np.random.randn(num_steps) * 0.05,
        'log_var_genus': np.linspace(0, 0.8, num_steps),
        'log_var_health': np.linspace(0, -0.5, num_steps),
        'lr': list(np.linspace(0, 6e-5, warmup_steps)) + list(np.linspace(6e-5, 1e-7, num_steps - warmup_steps))
    }
    
    print("\n1. Plotting Composite Losses...")
    plot_composite_losses(mock_log_history)
    
    print("\n2. Plotting Individual Task Losses...")
    plot_individual_task_losses(mock_log_history)
    
    print("\n3. Plotting Uncertainty Weights...")
    plot_uncertainty_weights(mock_log_history)
    
    print("\n4. Plotting Learning Rate Schedule...")
    plot_learning_rate(mock_log_history, warmup_steps=warmup_steps)