# Create file: tests/coral_mtl/metrics/test_three_tier_metric_pipeline.py
import json
import math
from pathlib import Path
from typing import Any, Dict, Generator

import pytest
import torch

from coral_mtl.metrics.metrics import CoralMTLMetrics
from coral_mtl.metrics.metrics_storer import (
    AdvancedMetricsProcessor,
    AsyncMetricsStorer,
    MetricsStorer,
)

# --- Test Suite for the Full Integrated Metrics Pipeline ---

@pytest.mark.slow
@pytest.mark.gpu
class TestThreeTierMetricPipeline:
    """
    A comprehensive integration test to stress the entire three-tier metrics pipeline.
    It verifies that the GPU-based Tier 1 accumulators, the asynchronous Tier 1.5
    storer, and the multi-process Tier 2 processor can all run concurrently
    without deadlocks, data loss, or race conditions.
    """

    @pytest.fixture
    def pipeline_components(self, splitter_mtl, device, tmp_path: Path) -> Generator[Dict[str, Any], None, None]:
        """
        ARRANGE: Sets up all real components for the three-tier pipeline and
        handles their graceful shutdown in the teardown phase.
        """
        # Tier 1 storer + async bridge
        metrics_storer = MetricsStorer(output_dir=str(tmp_path))
        metrics_storer.open_for_run(is_testing=False)

        # Tier 1: The main orchestrator for GPU metrics, connected to the async storer
        metrics_calculator = CoralMTLMetrics(
            splitter=splitter_mtl,
            storer=metrics_storer,
            device=device,
            ignore_index=255,
            use_async_storage=True
        )

        # Tier 1.5: The async storer for per-image CMs (thread-based)
        async_storer: AsyncMetricsStorer = metrics_calculator.async_storer
        if async_storer is None:
            raise RuntimeError("Expected CoralMTLMetrics to create an AsyncMetricsStorer")
        async_storer.open_for_run(is_testing=False)

        # Tier 2: The processor for advanced CPU-based metrics (process-based)
        processor = AdvancedMetricsProcessor(output_dir=str(tmp_path), num_cpu_workers=4)

        components = {
            "calculator": metrics_calculator,
            "storer": async_storer,
            "processor": processor
        }

        try:
            yield components
        finally:
            # TEARDOWN: Ensure all components are shut down cleanly.
            if processor._active:
                processor.shutdown()
            # The async storer is shut down by the metrics calculator's storer reference,
            # but an explicit shutdown is safe.
            async_storer.shutdown()
            metrics_storer.close()

    @pytest.fixture
    def synthetic_batch_generator(self, splitter_mtl, device) -> Generator[Dict[str, Any], None, None]:
        """
        A generator that mimics a DataLoader, yielding a stream of synthetic batches
        to simulate a full validation epoch.
        """
        def _generator():
            num_batches = 10
            batch_size = 4
            h, w = 64, 64

            for i in range(num_batches):
                original_mask = torch.zeros(batch_size, h, w, dtype=torch.long, device=device)
                original_mask[:, h // 4:h * 3 // 4, w // 4:w * 3 // 4] = (i % 5) + 1  # Use different real class IDs

                logits_dict = {}
                for task, details in splitter_mtl.hierarchical_definitions.items():
                    num_classes = len(details['ungrouped']['id2label'])
                    mapping = torch.from_numpy(details['ungrouped']['mapping_array']).to(device)
                    task_target = mapping[original_mask]

                    task_logits = torch.randn(batch_size, num_classes, h, w, device=device) * 0.1
                    task_logits.scatter_(1, task_target.unsqueeze(1), 10.0)
                    logits_dict[task] = task_logits

                yield {
                    'original_mask': original_mask,
                    'image_id': [f"img_batch{i}_item{j}" for j in range(batch_size)],
                    'predictions_logits': logits_dict,
                }

        return _generator()

    def test_full_pipeline_stress_test(
        self, pipeline_components: Dict[str, Any], synthetic_batch_generator
    ):
        """
        Simulates a full validation run, activating all three tiers of the metrics
        system, and then verifies the correctness of all generated output artifacts.
        """
        # --- 1. UNPACK and START components ---
        calculator: CoralMTLMetrics = pipeline_components["calculator"]
        storer: AsyncMetricsStorer = pipeline_components["storer"]
        processor: AdvancedMetricsProcessor = pipeline_components["processor"]
        
        processor.start()
        calculator.reset()
        
        # --- 2. ACT: Simulate a full validation loop under load ---
        total_images_processed = 0
        
        for batch in synthetic_batch_generator:
            batch_size = len(batch['image_id'])
            total_images_processed += batch_size
            
            # This single call triggers both Tier 1 and Tier 1.5 operations:
            # - Tier 1: GPU accumulators are updated.
            # - Tier 1.5: Per-image CMs are dispatched to the async storer's thread queue.
            calculator.update(
                predictions=None,
                original_targets=batch['original_mask'],
                image_ids=batch['image_id'],
                epoch=1,
                predictions_logits=batch['predictions_logits'],
                store_per_image=True
            )

            # This mimics the Trainer's role in dispatching jobs to Tier 2.
            first_task_logits = next(iter(batch['predictions_logits'].values()))
            pred_masks = torch.argmax(first_task_logits, dim=1)
            for i in range(batch_size):
                processor.dispatch_image_job(
                    image_id=batch['image_id'][i],
                    pred_mask_tensor=pred_masks[i],
                    target_mask_tensor=batch['original_mask'][i]
                )

        # --- 3. SHUTDOWN: Wait for all concurrent operations to complete ---
        tier1_report = calculator.compute()
        storer.wait_for_completion()
        processor.shutdown()

        # --- 4. ASSERT: Verify the outputs of all three tiers ---
        tmp_path = Path(storer.output_dir)

        # Tier 1 Verification: Check the final aggregated report
        assert "optimization_metrics" in tier1_report
        global_miou = tier1_report['optimization_metrics']['global.mIoU']
        boundary_f1 = tier1_report['optimization_metrics']['global.Boundary_F1']

        assert isinstance(global_miou, float)
        assert isinstance(boundary_f1, float)
        assert not math.isnan(global_miou)
        assert not math.isnan(boundary_f1)
        assert 0.0 <= global_miou <= 1.0
        assert 0.0 <= boundary_f1 <= 1.0
        
        # Tier 1.5 Verification: Check the async storer's JSONL output
        tier1_5_output_file = tmp_path / "validation_cms.jsonl"
        assert tier1_5_output_file.exists(), "Async storer (Tier 1.5) did not create its output file."
        with open(tier1_5_output_file, 'r') as f:
            tier1_5_lines = f.readlines()
        assert len(tier1_5_lines) == total_images_processed
        
        record1_5 = json.loads(tier1_5_lines[0])
        assert "image_id" in record1_5
        assert "per_image_metrics" in record1_5
        per_image_metrics = record1_5["per_image_metrics"]
        assert isinstance(per_image_metrics, dict)
        assert "global" in per_image_metrics
        global_per_image = per_image_metrics["global"]
        assert isinstance(global_per_image, dict)
        assert "mIoU" in global_per_image
        assert not math.isnan(global_per_image["mIoU"])

        # Tier 2 Verification: Check the advanced processor's JSONL output
        tier2_output_file = tmp_path / "advanced_metrics.jsonl"
        assert tier2_output_file.exists(), "Advanced processor (Tier 2) did not create its output file."
        with open(tier2_output_file, 'r') as f:
            tier2_lines = f.readlines()
        assert len(tier2_lines) == total_images_processed
        
        record2 = json.loads(tier2_lines[0])
        assert "image_id" in record2
        # Assert that the default set of advanced metrics were computed and stored
        assert "ASSD" in record2
        assert "HD95" in record2
        assert "ARI" in record2
        assert "PanopticQuality" in record2