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
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)

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
        print(f"Loading best model checkpoint from: {self.config.checkpoint_path}")
        self.model.load_state_dict(torch.load(self.config.checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # --- Step 2: Initialize Sliding Window Inferrer ---
        inferrer = SlidingWindowInferrer(
            model=self.model,
            patch_size=self.config.patch_size,
            stride=self.config.inference_stride,
            device=self.device,
            batch_size=self.config.inference_batch_size
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

                    # Perform sliding window inference on each image in the batch
                    # Note: We need to loop through each image since predict takes single images
                    batch_predictions = {}
                    for idx, single_image in enumerate(batch_images):
                        single_predictions = inferrer.predict(single_image)
                        if idx == 0:
                            # Initialize batch_predictions with the right structure
                            for task_name in single_predictions:
                                batch_predictions[task_name] = []
                        for task_name, logits in single_predictions.items():
                            batch_predictions[task_name].append(logits)
                    
                    # Stack predictions for batch processing
                    stitched_predictions_logits = {
                        task_name: torch.stack(task_logits, dim=0)
                        for task_name, task_logits in batch_predictions.items()
                    }

                    # Handle different model types for metrics calculator
                    # MTL models expect dictionary, baseline models expect single tensor
                    if len(stitched_predictions_logits) == 1 and 'segmentation' in stitched_predictions_logits:
                        # This is a baseline model - extract the single tensor for CoralMetrics
                        predictions_for_metrics = stitched_predictions_logits['segmentation']
                    else:
                        # This is an MTL model - pass the full dictionary for CoralMTLMetrics
                        predictions_for_metrics = stitched_predictions_logits

                    # Update metrics calculator with the required data
                    self.metrics_calculator.update(
                        predictions=predictions_for_metrics,
                        original_targets=original_masks,
                        image_ids=image_ids,
                        epoch=0,  # Test evaluation doesn't have epochs
                        store_per_image=True  # Enable per-image storage for test evaluation
                    )

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