# engine.py

import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn, scaler, device, log_history):
    """
    Executes a single training epoch according to the specified workflow.
    Handles forward pass, loss calculation, backpropagation, and metric logging.
    """
    model.train()
    
    loop = tqdm(dataloader, desc=f"Training Epoch", leave=True)
    for batch in loop:
        images = batch['image'].to(device, non_blocking=True)
        masks = {k: v.to(device, non_blocking=True) for k, v in batch['masks'].items()}

        # 1. Forward Pass with Automatic Mixed Precision (AMP)
        with torch.cuda.amp.autocast():
            predictions = model(images)
            loss_dict = loss_fn(predictions, masks)
            total_loss = loss_dict['total_loss']
        
        # 2. Backpropagation using GradScaler to prevent underflow
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 3. Update Learning Rate Scheduler
        scheduler.step()

        # 4. Log batch-level metrics for live monitoring
        log_history['total_loss'].append(total_loss.item())
        log_history['primary_loss'].append(loss_dict['primary_loss'].item())
        log_history['auxiliary_loss'].append(loss_dict['auxiliary_loss'].item())
        log_history['genus_loss'].append(loss_dict['genus_loss'].item())
        log_history['health_loss'].append(loss_dict['health_loss'].item())
        log_history['lr'].append(scheduler.get_last_lr()[0])
        log_history['log_var_genus'].append(loss_dict['log_var_genus'].item())
        log_history['log_var_health'].append(loss_dict['log_var_health'].item())
        
        loop.set_postfix(loss=total_loss.item())

def validate_one_epoch(model, dataloader, loss_fn, metrics_calculator, device):
    """
    Executes a single validation epoch.
    Performs inference, calculates validation loss, and updates metrics.
    """
    model.eval()
    total_val_loss = 0.0
    
    loop = tqdm(dataloader, desc=f"Validation", leave=True)
    with torch.no_grad():
        for batch in loop:
            images = batch['image'].to(device, non_blocking=True)
            masks = {k: v.to(device, non_blocking=True) for k, v in batch['masks'].items()}

            # Forward pass in inference mode
            with torch.cuda.amp.autocast():
                predictions = model(images)
                loss_dict = loss_fn(predictions, masks)
                total_val_loss += loss_dict['total_loss'].item()
            
            # Update the metrics calculator with the batch results
            metrics_calculator.update(predictions, masks)

    avg_val_loss = total_val_loss / len(dataloader)
    return avg_val_loss