# File: train_baseline.py
# Description: A self-contained script to train a single-task SegFormer baseline
#              on the original 39-class Coralscapes problem. This version includes
#              full integration with plotting and reporting utilities.

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import math
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# --- REUSABLE COMPONENTS (Assumed to be in the project structure) ---
from segformer_encoder import SegFormerEncoder
from augmentations import SegmentationAugmentation
from optimizer import create_optimizer_and_scheduler
from losses import FocalLoss, DiceLoss
from metrics import EPS
import live_monitoring  # Import for live training plots
import reporting       # Import for final performance reports
from huggingface_hub import hf_hub_download


# === 1. BASELINE-SPECIFIC CONFIGURATION ===

class ConfigBaseline:
    """Configuration for the single-task baseline experiment."""
    # --- Dataset and DataLoader ---
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PATCH_SIZE = 512

    # --- Model Architecture ---
    ENCODER_NAME = "nvidia/mit-b2"
    DECODER_CHANNEL = 768
    NUM_CLASSES = 39 + 1 # 39 classes + 1 'unlabeled' class

    # --- Optimizer and Scheduler ---
    LEARNING_RATE = 6e-5
    WEIGHT_DECAY = 0.01
    ADAM_BETAS = (0.9, 0.999)
    NUM_EPOCHS = 50
    WARMUP_STEPS_RATIO = 0.1

    # --- Loss Function ---
    HYBRID_ALPHA = 0.5
    FOCAL_GAMMA = 2.0

    # --- Training ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    OUTPUT_DIR = "training_outputs/baseline"
    BEST_MODEL_NAME = "best_baseline_model.pth"
    DEBUG = False


# === 2. HELPER FUNCTIONS & DATA HANDLING ===

def get_class_names() -> List[str]:
    """Downloads and loads the class names from the HF Hub."""
    repo_id = "EPFL-ECEO/coralscapes"
    id2label_path = hf_hub_download(repo_id=repo_id, filename="id2label.json", repo_type="dataset")
    with open(id2label_path, "r") as f:
        id2label_str_keys = json.load(f)
    id2label = {int(k): v for k, v in id2label_str_keys.items()}
    id2label[0] = "unlabeled"
    return [id2label[i] for i in range(len(id2label))]

class CoralscapesSingleTaskDataset(Dataset):
    """Simplified dataset for the single-task baseline."""
    def __init__(self,
                 hf_dataset: 'datasets.Dataset',
                 split: str = 'train',
                 augmentations: Optional[SegmentationAugmentation] = None,
                 patch_size: int = 512):
        self.dataset_split = hf_dataset[split]
        self.augmentations = augmentations
        self.patch_size = patch_size
        if self.augmentations is None:
            from torchvision.transforms import v2
            self.default_transform = v2.Compose([
                v2.Resize((self.patch_size, self.patch_size), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.dataset_split)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.dataset_split[idx]
        image, mask = example['image'], example['label']
        if self.augmentations:
            final_image, final_masks_dict = self.augmentations(image, {'mask': mask})
            final_mask = final_masks_dict['mask']
        else:
            from torchvision.transforms import v2
            final_image = self.default_transform(image)
            resized_mask = v2.functional.resize(mask, (self.patch_size, self.patch_size), interpolation=v2.InterpolationMode.NEAREST)
            final_mask = torch.from_numpy(np.array(resized_mask)).long()
        return {'image': final_image, 'mask': final_mask}


# === 3. BASELINE-SPECIFIC MODEL ARCHITECTURE ===

class SegFormerMLPDecoder(nn.Module):
    """A simple MLP decoder head, standard for SegFormer."""
    def __init__(self, encoder_channels: List[int], decoder_channel: int, num_classes: int):
        super().__init__()
        self.mlps = nn.ModuleList([nn.Sequential(nn.Conv2d(c, decoder_channel, 1), nn.BatchNorm2d(decoder_channel), nn.ReLU(inplace=True)) for c in encoder_channels])
        self.fusion = nn.Sequential(nn.Conv2d(decoder_channel * len(encoder_channels), decoder_channel, 1, bias=False), nn.BatchNorm2d(decoder_channel), nn.ReLU(inplace=True))
        self.predictor = nn.Conv2d(decoder_channel, num_classes, 1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        projected = [F.interpolate(self.mlps[i](feat), size=target_size, mode='bilinear', align_corners=False) for i, feat in enumerate(features)]
        fused = self.fusion(torch.cat(projected, dim=1))
        return self.predictor(fused)

class SegFormerSingleTask(nn.Module):
    """A single-task SegFormer model for the baseline."""
    def __init__(self, encoder_name: str, decoder_channel: int, num_classes: int):
        super().__init__()
        self.encoder = SegFormerEncoder(pretrained_weights_path=encoder_name)
        self.decoder = SegFormerMLPDecoder(self.encoder.channels, decoder_channel, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        logits_quarter_res = self.decoder(features)
        return F.interpolate(logits_quarter_res, size=images.shape[-2:], mode='bilinear', align_corners=False)


# === 4. BASELINE-SPECIFIC LOSS & METRICS ===

class BaselineHybridLoss(nn.Module):
    """A simple hybrid Focal + Dice loss for the single-task baseline."""
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.focal_loss = FocalLoss(gamma=gamma, ignore_index=0)
        self.dice_loss = DiceLoss(ignore_index=0)

    def forward(self, logits, targets):
        return self.alpha * self.focal_loss(logits, targets) + (1 - self.alpha) * self.dice_loss(logits, targets)

class SingleTaskMetrics:
    """A simplified metrics calculator for the single-task baseline."""
    def __init__(self, num_classes: int, device: torch.device):
        self.num_classes = num_classes
        self.device = device
        self.ignore_index = 0
        self.reset()

    def reset(self):
        self.cm = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device=self.device)

    def update(self, preds_logits: torch.Tensor, targets: torch.Tensor):
        preds = torch.argmax(preds_logits, dim=1)
        mask = (targets != self.ignore_index)
        cm_update = torch.bincount(
            self.num_classes * targets[mask].long() + preds[mask].long(),
            minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)
        self.cm += cm_update.to(self.device)

    def compute(self) -> Dict[str, any]:
        cm_np = self.cm.cpu().numpy()
        tp = np.diag(cm_np)
        fp = cm_np.sum(axis=0) - tp
        fn = cm_np.sum(axis=1) - tp
        iou = tp / (tp + fp + fn + EPS)
        miou = np.nanmean(iou[1:]) # Exclude 'unlabeled' class 0 from mean
        return {'mIoU': miou, 'per_class_iou': iou}


# === 5. BASELINE-SPECIFIC TRAINING & VALIDATION ENGINE ===

def train_one_epoch_baseline(model, dataloader, optimizer, scheduler, loss_fn, scaler, device, grad_accumulation_steps, log_history):
    model.train()
    loop = tqdm(enumerate(dataloader), desc="Training", total=len(dataloader), leave=True)
    optimizer.zero_grad()
    for i, batch in loop:
        images, masks = batch['image'].to(device, non_blocking=True), batch['mask'].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=str(device), dtype=torch.float16):
            loss = loss_fn(model(images), masks) / grad_accumulation_steps
        scaler.scale(loss).backward()
        if (i + 1) % grad_accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        log_history['train_loss'].append(loss.item() * grad_accumulation_steps)
        log_history['lr'].append(scheduler.get_last_lr()[0])
        loop.set_postfix(loss=(loss.item() * grad_accumulation_steps))

def validate_one_epoch_baseline(model, dataloader, loss_fn, metrics_calculator, device):
    model.eval()
    total_val_loss = 0.0
    loop = tqdm(dataloader, desc="Validation", leave=True)
    with torch.no_grad():
        for batch in loop:
            images, masks = batch['image'].to(device, non_blocking=True), batch['mask'].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=str(device), dtype=torch.float16):
                total_val_loss += loss_fn(model(images), masks).item()
            metrics_calculator.update(model(images), masks)
    return total_val_loss / len(dataloader)


# === 6. MAIN EXECUTION SCRIPT ===

def main():
    config = ConfigBaseline()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    class_names = get_class_names()
    hf_dataset = load_dataset("EPFL-ECEO/coralscapes")
    
    if config.DEBUG:
        from datasets import DatasetDict
        train_subset = hf_dataset['train'].select(range(config.BATCH_SIZE * 2))
        val_subset = hf_dataset['validation'].select(range(config.BATCH_SIZE * 2))
        hf_dataset = DatasetDict({'train': train_subset, 'validation': val_subset})
    
    train_augs = SegmentationAugmentation(patch_size=config.PATCH_SIZE)
    train_dataset = CoralscapesSingleTaskDataset(hf_dataset, 'train', train_augs, config.PATCH_SIZE)
    val_dataset = CoralscapesSingleTaskDataset(hf_dataset, 'validation', None, config.PATCH_SIZE)
    
    train_loader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    
    model = SegFormerSingleTask(config.ENCODER_NAME, config.DECODER_CHANNEL, config.NUM_CLASSES).to(config.DEVICE)
    loss_fn = BaselineHybridLoss(alpha=config.HYBRID_ALPHA, gamma=config.FOCAL_GAMMA).to(config.DEVICE)

    num_steps_per_epoch = math.ceil(len(train_loader) / config.GRADIENT_ACCUMULATION_STEPS)
    total_steps = num_steps_per_epoch * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_STEPS_RATIO)
    
    optimizer, scheduler = create_optimizer_and_scheduler(model, config.LEARNING_RATE, config.WEIGHT_DECAY, config.ADAM_BETAS, total_steps, warmup_steps)
    
    scaler = torch.amp.GradScaler()
    metrics_calculator = SingleTaskMetrics(config.NUM_CLASSES, config.DEVICE)
    
    best_miou = -1.0
    log_history = defaultdict(list)
    validation_log_history = defaultdict(list)

    print(f"--- Starting Baseline Training for {config.NUM_EPOCHS} epochs ---")
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{config.NUM_EPOCHS} =====")
        train_one_epoch_baseline(model, train_loader, optimizer, scheduler, loss_fn, scaler, config.DEVICE, config.GRADIENT_ACCUMULATION_STEPS, log_history)
        avg_val_loss = validate_one_epoch_baseline(model, val_loader, loss_fn, metrics_calculator, config.DEVICE)
        val_metrics = metrics_calculator.compute()
        
        validation_log_history['val_loss'].append(avg_val_loss)
        validation_log_history['mIoU'].append(val_metrics['mIoU'])
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Avg Validation Loss: {avg_val_loss:.4f}")
        print(f"  Validation mIoU: {val_metrics['mIoU']:.4f} (Best: {max(best_miou, val_metrics['mIoU']):.4f})")
        
        if val_metrics['mIoU'] > best_miou:
            best_miou = val_metrics['mIoU']
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, config.BEST_MODEL_NAME))
            print(f"  >>> New best model saved.")
            
        metrics_calculator.reset()
        
    print("\n--- Baseline Training Complete ---")
    
    # --- POST-TRAINING ANALYSIS AND PLOTTING ---
    print("\n--- Generating final plots and reports ---")
    
    # 1. Live Monitoring Plots
    live_monitoring.plot_learning_rate(log_history, warmup_steps, save_path=f"{config.OUTPUT_DIR}/baseline_lr_schedule.png")
    
    # Simple loss plot (train only)
    plt.figure(figsize=(12, 7))
    plt.plot(log_history['train_loss'], label='Training Loss')
    plt.title('Training Loss per Step')
    plt.xlabel('Training Steps'); plt.ylabel('Loss'); plt.yscale('log'); plt.legend(); plt.grid(True)
    plt.savefig(f"{config.OUTPUT_DIR}/baseline_train_loss.png")
    plt.show()

    # 2. Load best model for final evaluation
    best_model_path = os.path.join(config.OUTPUT_DIR, config.BEST_MODEL_NAME)
    model.load_state_dict(torch.load(best_model_path))
    
    # 3. Final Performance Plots
    # mIoU vs. Epoch
    epochs = range(1, config.NUM_EPOCHS + 1)
    best_epoch_idx = np.argmax(validation_log_history['mIoU'])
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, validation_log_history['mIoU'], marker='o', label=f"Validation mIoU (Best: {best_miou:.4f})")
    plt.axvline(x=best_epoch_idx + 1, color='r', linestyle='--', label=f'Best Epoch: {best_epoch_idx+1}')
    plt.title('Baseline Validation mIoU per Epoch'); plt.xlabel('Epoch'); plt.ylabel('mIoU'); plt.legend(); plt.grid(True)
    plt.savefig(f"{config.OUTPUT_DIR}/baseline_miou_vs_epoch.png")
    plt.show()

    # Run one final validation pass to get metrics from the best model
    final_metrics_calc = SingleTaskMetrics(config.NUM_CLASSES, config.DEVICE)
    _ = validate_one_epoch_baseline(model, val_loader, loss_fn, final_metrics_calc, config.DEVICE)
    final_metrics = final_metrics_calc.compute()
    
    # Per-Class IoU Bar Chart
    # We exclude the 'unlabeled' class (index 0) from the plot for clarity
    class_iou_dict = {name: score for name, score in zip(class_names[1:], final_metrics['per_class_iou'][1:])}
    reporting.plot_per_class_iou_bar_chart(class_iou_dict, "Baseline", save_path=f"{config.OUTPUT_DIR}/baseline_per_class_iou.png")
    
    # Confusion Matrix
    # We also exclude the 'unlabeled' class from the CM plot
    cm_for_plot = final_metrics_calc.cm.cpu().numpy()[1:, 1:]
    reporting.plot_confusion_matrix(cm_for_plot, class_names[1:], "Baseline", save_path=f"{config.OUTPUT_DIR}/baseline_confusion_matrix.png")

    # 4. Qualitative Results Grid
    print("Generating qualitative results grid...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        images = batch['image'].to(config.DEVICE)
        gt_masks = batch['mask']
        pred_logits = model(images)
        pred_masks = torch.argmax(pred_logits, dim=1)

        # reporting.plot_qualitative_grid expects dicts, so we adapt
        reporting.plot_qualitative_grid(
            images.cpu(), 
            {'Ground Truth': gt_masks}, 
            {'Prediction': pred_masks.cpu()},
            save_path=f"{config.OUTPUT_DIR}/baseline_qualitative_grid.png"
        )

if __name__ == "__main__":
    main()