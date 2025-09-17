import torch
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

# Correctly import the required classes from the project structure
from ..model.core import CoralMTLModel
from .inference import SlidingWindowInferrer
from .metrics import AbstractCoralMetrics
from ..utils.visualization import Visualizer

class Evaluator:
    """
    Orchestrates the final, one-time evaluation of a trained model on a test dataset.

    This class implements the full testing pipeline described in Section 8.3 of the
    project specification. It loads the best model checkpoint, uses the
    SlidingWindowInferrer to generate predictions on full-resolution images,
    computes a comprehensive set of metrics, and saves all evaluation artifacts.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 test_loader: torch.utils.data.DataLoader,
                 metrics_calculator: AbstractCoralMetrics,
                 config: object):
        """
        Initializes the Evaluator.

        Args:
            model (torch.nn.Module): The model architecture (weights will be loaded later).
            test_loader (torch.utils.data.DataLoader): The data loader for the test set,
                which yields full-resolution images and their corresponding masks.
            metrics_calculator (AbstractCoralMetrics): An initialized instance of a metrics
                calculator (e.g., CoralMTLMetrics or CoralMetrics).
            config (object): A configuration object containing paths and parameters, such as
                DEVICE, CHECKPOINT_PATH, OUTPUT_DIR, PATCH_SIZE, PRIMARY_TASKS etc.
        """
        self.model = model
        self.test_loader = test_loader
        self.metrics_calculator = metrics_calculator
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.output_dir = Path(config.OUTPUT_DIR)

    def evaluate(self) -> Dict[str, float]:
        """
        Executes the entire testing pipeline on the test dataset.

        Returns:
            Dict[str, float]: A dictionary containing the final computed metrics.
        """
        # --- Step 1: Load Best Model (Spec Section 8.3) ---
        print(f"Loading best model checkpoint from: {self.config.CHECKPOINT_PATH}")
        self.model.load_state_dict(torch.load(self.config.CHECKPOINT_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # --- Step 2: Initialize Sliding Window Inferrer (Spec Section 8.3) ---
        inferrer = SlidingWindowInferrer(
            model=self.model,
            patch_size=self.config.PATCH_SIZE,
            stride=self.config.INFERENCE_STRIDE,
            device=self.device,
            batch_size=self.config.INFERENCE_BATCH_SIZE
        )

        # --- Step 3: Evaluation Loop (Spec Section 8.3) ---
        self.metrics_calculator.reset()
        qualitative_samples = []

        loop = tqdm(self.test_loader, desc="Evaluating on Test Set")
        with torch.no_grad():
            for batch in loop:
                # The test loader yields full images, typically with a batch size of 1.
                for i in range(batch['image'].shape[0]):
                    full_image = batch['image'][i]  # Shape: (C, H, W)
                    
                    # Perform sliding window inference to get stitched prediction logits
                    stitched_predictions_logits = inferrer.predict(full_image)

                    # Update metrics with full-resolution stitched maps and ground truth
                    is_mtl = 'masks' in batch
                    if is_mtl:
                        gt_masks = {k: v[i].unsqueeze(0).to(self.device) for k, v in batch['masks'].items()}
                        pred_logits_batched = {k: v.unsqueeze(0) for k, v in stitched_predictions_logits.items()}
                    else: # Baseline (single task) case
                        gt_masks = batch['mask'][i].unsqueeze(0).to(self.device)
                        # Correctly extract the single tensor from the inferrer's output dict
                        pred_logits_batched = stitched_predictions_logits['segmentation'].unsqueeze(0)

                    self.metrics_calculator.update(pred_logits_batched, gt_masks)

                    # Collect samples for qualitative visualization
                    if len(qualitative_samples) < self.config.NUM_VISUALIZATIONS:
                        qualitative_samples.append({
                            'image': full_image.cpu(),
                            'predictions': {k: v.cpu() for k, v in stitched_predictions_logits.items()},
                            'ground_truth': {k: v[i].cpu() for k, v in batch['masks'].items()} if is_mtl else batch['mask'][i].cpu()
                        })

        # --- Step 4: Final Metric Computation and Reporting (Spec Section 8.3) ---
        final_metrics = self.metrics_calculator.compute()
        
        print("\n--- Final Test Set Results ---")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        self._save_results(final_metrics)
        self._save_qualitative_visualizations(qualitative_samples)
        
        return final_metrics

    def _save_results(self, metrics: Dict[str, float]):
        """Saves the final metrics dictionary to a JSON file."""
        output_path = self.output_dir / "test_metrics.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Final metrics saved to {output_path}")

    def _save_qualitative_visualizations(self, samples: List[Dict]):
        """Generates and saves qualitative result images using the Visualizer class."""
        if not samples:
            return

        visualizer = Visualizer(output_dir=str(self.output_dir))

        # Prepare batched tensors for the visualizer
        images_batch = torch.stack([s['image'] for s in samples])
        
        # Visualize each primary task separately
        for task_name in self.config.PRIMARY_TASKS:
            # Check if task data is available in the samples
            if task_name not in samples[0]['predictions']:
                print(f"Warning: Task '{task_name}' not found in predictions for visualization.")
                continue

            gt_masks_batch = torch.stack([s['ground_truth'][task_name] for s in samples if isinstance(s['ground_truth'], dict)])
            
            # If ground_truth is not a dict (baseline case), it might apply to the first primary task
            if len(gt_masks_batch) == 0 and task_name == self.config.PRIMARY_TASKS[0]:
                 gt_masks_batch = torch.stack([s['ground_truth'] for s in samples if not isinstance(s['ground_truth'], dict)])
            
            if len(gt_masks_batch) == 0:
                print(f"Warning: No ground truth masks found for task '{task_name}' to visualize.")
                continue

            pred_logits_batch = torch.stack([s['predictions'][task_name] for s in samples])
            pred_masks_batch = torch.argmax(pred_logits_batch, dim=1)
            
            output_filename = f"qualitative_test_results_{task_name}.png"
            visualizer.plot_qualitative_results(
                images=images_batch,
                gt_masks=gt_masks_batch,
                pred_masks=pred_masks_batch,
                task_name=task_name.capitalize(),
                filename=output_filename,
                num_samples=len(samples)
            )
            print(f"Qualitative results for '{task_name}' saved to {self.output_dir / output_filename}")