import torch
import json
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List


from .inference import SlidingWindowInferrer
from .metrics import AbstractCoralMetrics
from coral_mtl.utils.metrics_storer import MetricsStorer

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
                 metrics_storer: MetricsStorer,
                 config: object):
        """
        Initializes the Evaluator.

        Args:
            model (torch.nn.Module): The model architecture (weights will be loaded later).
            test_loader (torch.utils.data.DataLoader): The data loader for the test set,
                which yields full-resolution images and their corresponding masks.
            metrics_calculator (AbstractCoralMetrics): An initialized instance of a metrics
                calculator (e.g., CoralMTLMetrics or CoralMetrics).
            metrics_storer (MetricsStorer): An initialized MetricsStorer for saving metrics.
            config (object): A configuration object containing paths and parameters, such as
                DEVICE, CHECKPOINT_PATH, OUTPUT_DIR, PATCH_SIZE, PRIMARY_TASKS etc.
        """
        self.model = model
        self.test_loader = test_loader
        self.metrics_calculator = metrics_calculator
        self.metrics_storer = metrics_storer
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.output_dir = Path(config.OUTPUT_DIR)

    def evaluate(self) -> Dict[str, Any]:
        """
        Executes the entire testing pipeline on the test dataset.

        This method loads the best model, performs sliding window inference on each
        test image, updates the metrics calculator with raw predictions and original
        ground truth, stores per-image confusion matrices, and finally computes
        and saves a comprehensive final evaluation report.

        Returns:
            Dict[str, Any]: The final, nested dictionary of computed metrics.
        """
        # --- Step 1: Load Best Model ---
        print(f"Loading best model checkpoint from: {self.config.CHECKPOINT_PATH}")
        self.model.load_state_dict(torch.load(self.config.CHECKPOINT_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # --- Step 2: Initialize Sliding Window Inferrer ---
        inferrer = SlidingWindowInferrer(
            model=self.model,
            patch_size=self.config.PATCH_SIZE,
            stride=self.config.INFERENCE_STRIDE,
            device=self.device,
            batch_size=self.config.INFERENCE_BATCH_SIZE
        )

        # --- Step 3: Evaluation Loop ---
        self.metrics_calculator.reset()
        
        try:
            # Open file handle for the duration of the evaluation run
            self.metrics_storer.open_for_run(is_testing=True)
            loop = tqdm(self.test_loader, desc="Evaluating on Test Set")
            
            with torch.no_grad():
                for batch in loop:
                    # The test loader yields batches of full images.
                    # The `original_mask` and `image_id` are now essential parts of the batch.
                    batch_images = batch['image']
                    original_masks = batch['original_mask'].to(self.device)
                    image_ids = batch['image_id']

                    # Perform sliding window inference on the entire batch of images
                    # (Assuming the inferrer can handle a batch of full-size images)
                    stitched_predictions_logits = inferrer.predict_batch(batch_images)

                    # Update metrics calculator with the required data
                    self.metrics_calculator.update(
                        predictions=stitched_predictions_logits,
                        original_targets=original_masks,
                        image_ids=image_ids
                    )
                    
                    # After updating, the calculator's buffer holds the per-image CMs for this batch.
                    # We store them immediately to keep memory usage low.
                    for img_id, cms in self.metrics_calculator.per_image_cms_buffer:
                        self.metrics_storer.store_per_image_cms(img_id, cms, is_testing=True)
                    
                    # Clear the buffer after storing
                    self.metrics_calculator.per_image_cms_buffer.clear()

        finally:
            # Ensure the file handle is always closed, even if errors occur
            self.metrics_storer.close()

        # --- Step 4: Final Metric Computation and Reporting ---
        print("\nComputing final metrics from aggregated confusion matrices...")
        final_metrics_report = self.metrics_calculator.compute()
        
        # Save the full, detailed report using the storer
        self.metrics_storer.save_final_report(final_metrics_report, "test_metrics_full_report.json")
        print(f"Full evaluation report saved to {self.output_dir / 'test_metrics_full_report.json'}")
        print(f"Per-image confusion matrices saved to {self.output_dir / 'test_cms.jsonl'}")

        # Print a high-level summary to the console for quick review
        print("\n--- Final Test Set Summary ---")
        optim_metrics = final_metrics_report.get('optimization_metrics', {})
        for key, value in optim_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        return final_metrics_report