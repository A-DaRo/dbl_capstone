# Create file: tests/coral_mtl/metrics/metrics_storer/test_advanced_metrics_processor.py
import json
import multiprocessing
import queue
import sys
import time
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from coral_mtl.metrics.metrics_storer import AdvancedMetricsProcessor

# --- Test Class for AdvancedMetricsProcessor ---

class TestAdvancedMetricsProcessor:
    """
    Extensive test cases for the multi-process AdvancedMetricsProcessor.
    These tests verify its complex lifecycle, concurrency, data integrity,
    and robustness to configuration and environment issues.
    """

    @pytest.fixture
    def processor(self, tmp_path: Path) -> Iterator[AdvancedMetricsProcessor]:
        """
        Provides an initialized AdvancedMetricsProcessor and ensures its
        shutdown to prevent leaking processes.
        """
        # Use a small number of workers for testing efficiency
        proc_instance = AdvancedMetricsProcessor(output_dir=str(tmp_path), num_cpu_workers=2)
        yield proc_instance
        # Crucial cleanup: ensure processes are terminated after each test.
        if proc_instance._active:
            proc_instance.shutdown()

    # --- Lifecycle and State Management Tests ---

    def test_lifecycle_management(self, processor: AdvancedMetricsProcessor):
        """Verify start() and shutdown() manage processes correctly."""
        processor.start()
        assert processor._active
        assert len(processor.worker_pool) == processor.num_cpu_workers
        assert processor.io_writer_process is not None

        # idempotent
        processor.start()
        assert len(processor.worker_pool) == processor.num_cpu_workers

        processor.shutdown()
        assert not processor._active

    def test_restart_cycle_after_shutdown(self, tmp_path: Path):
        """Processor can be restarted after a clean shutdown and still process jobs."""
        processor = AdvancedMetricsProcessor(output_dir=str(tmp_path), num_cpu_workers=2)
        processor.start()
        processor.shutdown()

        # Second start should spin up fresh workers and write output successfully
        processor.start()
        pred_mask = np.zeros((8, 8), dtype=np.uint8)
        target_mask = np.zeros((8, 8), dtype=np.uint8)
        processor.dispatch_image_job("img_restart", pred_mask, target_mask)
        processor.shutdown()

        output_file = tmp_path / "advanced_metrics.jsonl"
        with output_file.open("r") as handle:
            record = json.loads(handle.readline())
        assert record["image_id"] == "img_restart"

    def test_shutdown_is_safe_when_not_started(self, processor: AdvancedMetricsProcessor):
        """Verify that calling shutdown() on a non-active processor is safe."""
        # This should execute without raising any errors.
        processor.shutdown()
        assert not processor._active

    # --- Concurrency and End-to-End Tests ---

    def test_end_to_end_stress_test(self, tmp_path: Path):
        """
        Dispatch many jobs and verify all are processed and written, testing
        the full pipeline for deadlocks and data loss.
        """
        num_jobs = 50
        # Use a higher worker count for a more realistic stress test
        processor = AdvancedMetricsProcessor(output_dir=str(tmp_path), num_cpu_workers=4)
        processor.start()
        
        pred_mask = np.zeros((32, 32), dtype=np.uint8)
        target_mask = np.zeros((32, 32), dtype=np.uint8)

        # Dispatch jobs in a tight loop
        for i in range(num_jobs):
            processor.dispatch_image_job(f"img_{i}", pred_mask, target_mask)
            
        # Shutdown blocks until all queues are processed and files are written
        processor.shutdown()

        # --- Verification ---
        output_file = tmp_path / "advanced_metrics.jsonl"
        assert output_file.exists(), "Output file was not created."
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == num_jobs, "The number of written records does not match the number of dispatched jobs."
        image_ids = {json.loads(line)['image_id'] for line in lines}
        assert len(image_ids) == num_jobs, "All dispatched image_ids should be present in the output."

    # --- Configuration and Robustness Tests ---

    def test_task_gating_writes_only_enabled_metrics(self, tmp_path: Path):
        """
        Verify that the output record only contains keys for the tasks
        explicitly enabled in the constructor.
        """
        # Enable only two specific metrics
        enabled_tasks = ["ARI", "ASSD"]
        processor = AdvancedMetricsProcessor(
            output_dir=str(tmp_path),
            num_cpu_workers=1,
            enabled_tasks=enabled_tasks
        )
        processor.start()
        
        pred_mask = np.array([[1, 1, 0], [0, 2, 2]], dtype=np.uint8)
        target_mask = np.array([[1, 0, 0], [2, 2, 2]], dtype=np.uint8)
        processor.dispatch_image_job("img_gated", pred_mask, target_mask)
        processor.shutdown()

        # --- Verification ---
        output_file = tmp_path / "advanced_metrics.jsonl"
        with open(output_file, 'r') as f:
            line = f.readline()
        record = json.loads(line)

        # Standard keys should always be present
        assert "image_id" in record
        assert "timestamp" in record

        # Check for enabled tasks
        for task in enabled_tasks:
            assert task in record, f"Enabled task '{task}' is missing from output."

        # Check that disabled tasks are NOT present
        assert "HD95" not in record, "Disabled task 'HD95' should not be in output."
        assert "PanopticQuality" not in record, "Disabled task 'PanopticQuality' should not be in output."

    def test_dispatch_requires_active_processor(self, processor: AdvancedMetricsProcessor):
        """Calling dispatch before start should raise a helpful error."""
        with pytest.raises(RuntimeError, match="not started"):
            processor.dispatch_image_job("img_fail", np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8))

    def test_dispatch_retries_when_queue_is_full(self, tmp_path: Path):
        """Queue Full exception should trigger a blocking retry."""
        processor = AdvancedMetricsProcessor(output_dir=str(tmp_path), num_cpu_workers=1)
        processor.job_queue = MagicMock()
        processor._active = True
        processor.job_queue.put.side_effect = [queue.Full(), None]

        processor.dispatch_image_job("img_retry", np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8))

        assert processor.job_queue.put.call_count == 2
        first_call_kwargs = processor.job_queue.put.call_args_list[0].kwargs
        second_call_kwargs = processor.job_queue.put.call_args_list[1].kwargs
        assert first_call_kwargs.get("timeout") == 5.0
        assert second_call_kwargs.get("timeout") is None

    def test_enabled_tasks_are_deduplicated_and_clamped(self, tmp_path: Path):
        """Constructor should clamp worker count and deduplicate enabled task list while preserving order."""
        processor = AdvancedMetricsProcessor(
            output_dir=str(tmp_path),
            num_cpu_workers=99,
            enabled_tasks=["ARI", "HD95", "ARI", "HD95", "PanopticQuality"],
        )
        assert processor.num_cpu_workers == 8
        assert processor._enabled_task_set == ["ARI", "HD95", "PanopticQuality"]

    def test_io_writer_filters_unrequested_metrics(self, tmp_path: Path):
        """The writer should persist only metrics explicitly enabled."""
        results_queue = multiprocessing.Queue()
        shutdown_event = multiprocessing.Event()

        writer_process = multiprocessing.Process(
            target=AdvancedMetricsProcessor._io_writer_loop,
            args=(results_queue, str(tmp_path), ["ARI"], shutdown_event),
        )
        writer_process.start()

        results_queue.put({"image_id": "img_writer", "timestamp": 123.0, "ARI": 0.77, "HD95": 9.9})
        results_queue.put(None)
        writer_process.join(timeout=10.0)
        assert not writer_process.is_alive()

        output_file = tmp_path / "advanced_metrics.jsonl"
        with output_file.open("r") as handle:
            record = json.loads(handle.readline())

        assert record == {"image_id": "img_writer", "timestamp": 123.0, "ARI": 0.77}
        results_queue.close()

    def test_io_writer_gracefully_handles_empty_queue_shutdown(self, tmp_path: Path):
        """Writer should terminate cleanly even if no payloads are ever enqueued."""
        results_queue = multiprocessing.Queue()
        shutdown_event = multiprocessing.Event()

        writer_process = multiprocessing.Process(
            target=AdvancedMetricsProcessor._io_writer_loop,
            args=(results_queue, str(tmp_path), ["ARI"], shutdown_event),
        )
        writer_process.start()

        results_queue.put(None)
        writer_process.join(timeout=10.0)
        assert not writer_process.is_alive()

        output_file = tmp_path / "advanced_metrics.jsonl"
        assert output_file.exists()
        with output_file.open("r") as handle:
            assert handle.read() == ""
        results_queue.close()

    @pytest.mark.optdeps
    def test_missing_dependency_is_handled_gracefully(self, tmp_path: Path):
        """
        Verify that a missing optional library (e.g., SimpleITK) does not crash
        the worker, which should skip the metric and process others.
        """
        # Simulate SimpleITK not being installed
        with patch.dict(sys.modules, {'SimpleITK': None}):
            processor = AdvancedMetricsProcessor(
                output_dir=str(tmp_path),
                num_cpu_workers=1,
                enabled_tasks=["ASSD", "HD95", "ARI"]
            )
            processor.start()
            try:
                pred_mask = np.ones((32, 32), dtype=np.uint8)
                target_mask = np.ones((32, 32), dtype=np.uint8)
                processor.dispatch_image_job("img_no_sitk", pred_mask, target_mask)
            finally:
                processor.shutdown()

        # --- Verification ---
        output_file = tmp_path / "advanced_metrics.jsonl"
        assert output_file.exists(), "Worker crashed, no output file created."
        with open(output_file, 'r') as f:
            record = json.loads(f.readline())
        
        assert record["image_id"] == "img_no_sitk"
        # The metric from the available library (sklearn) should be computed
        assert "ARI" in record
        assert record["ARI"] is not None
        
        # The metrics from the missing library should be absent or defaulted to 0.0
        assert record.get("ASSD", 0.0) == 0.0
        assert record.get("HD95", 0.0) == 0.0

    # --- Data Handling Tests ---

    @pytest.mark.parametrize("input_type", ["numpy", "torch_cpu", "torch_gpu"])
    def test_dispatch_handles_different_input_types(self, processor: AdvancedMetricsProcessor, input_type):
        """
        Verify that `dispatch_image_job` correctly handles NumPy arrays and
        PyTorch tensors from CPU or GPU.
        """
        # We only need to check that the job is put on the queue, not process it.
        # This makes the test faster and more focused.
        mask_data = np.array([[1, 0], [0, 1]], dtype=np.uint8)

        if input_type == "numpy":
            pred_mask = mask_data
            target_mask = mask_data
        elif input_type == "torch_cpu":
            pred_mask = torch.from_numpy(mask_data)
            target_mask = torch.from_numpy(mask_data)
        elif input_type == "torch_gpu":
            if not torch.cuda.is_available():
                pytest.skip("GPU not available for torch_gpu test")
            pred_mask = torch.from_numpy(mask_data).to("cuda")
            target_mask = torch.from_numpy(mask_data).to("cuda")

        # Manually prepare a lightweight queue to capture the dispatched job
        processor.job_queue = queue.Queue()
        processor._active = True

        try:
            processor.dispatch_image_job("img_types", pred_mask, target_mask)
            job = processor.job_queue.get_nowait()
        finally:
            processor._active = False
            processor.job_queue = None

        img_id, pred_np, target_np = job
        assert img_id == "img_types"
        assert isinstance(pred_np, np.ndarray)
        assert pred_np.dtype == np.uint8
        assert isinstance(target_np, np.ndarray)
        assert target_np.dtype == np.uint8