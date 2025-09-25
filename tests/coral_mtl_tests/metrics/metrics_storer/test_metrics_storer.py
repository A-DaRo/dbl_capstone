# Edit file: tests/coral_mtl/metrics/metrics_storer/test_advanced_metrics_processor.py
import json
import queue
import sys
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import numpy as np
import pytest
import torch

from coral_mtl.metrics.metrics_storer import AdvancedMetricsProcessor

class TestAdvancedMetricsProcessor:
    """
    Extensive test cases for the multi-process AdvancedMetricsProcessor, covering
    lifecycle, concurrency, data integrity, and robustness.
    """

    @pytest.fixture
    def processor(self, tmp_path: Path) -> Iterator[AdvancedMetricsProcessor]:
        """Provides an initialized processor and ensures shutdown."""
        proc_instance = AdvancedMetricsProcessor(output_dir=str(tmp_path), num_cpu_workers=2)
        yield proc_instance
        if proc_instance._active:
            proc_instance.shutdown()

    # --- Lifecycle and State Management Tests ---

    def test_lifecycle_management(self, processor: AdvancedMetricsProcessor):
        """Verify start() and shutdown() manage processes correctly."""
        processor.start()
        assert processor._active
        assert len(processor.worker_pool) == processor.num_cpu_workers
        assert processor.io_writer_process is not None

        processor.start()
        assert len(processor.worker_pool) == processor.num_cpu_workers

        processor.shutdown()
        assert not processor._active

    def test_repeated_start_shutdown_cycles(self, processor: AdvancedMetricsProcessor):
        """Test repeated start/shutdown cycles for robustness."""
        for _ in range(3):
            processor.start()
            assert processor._active
            processor.dispatch_image_job("img_cycle", np.zeros((4,4)), np.zeros((4,4)))
            processor.shutdown()
            assert not processor._active
        # Test passes if no errors are raised

    def test_zero_workers_clamped_to_one(self, tmp_path: Path):
        """A worker count of zero should be clamped to one for safety."""
        processor = AdvancedMetricsProcessor(output_dir=str(tmp_path), num_cpu_workers=0)
        assert processor.num_cpu_workers == 1

    # --- Concurrency, Stress, and Race Condition Tests ---

    def test_end_to_end_stress_test(self, tmp_path: Path):
        """
        Dispatch many jobs and verify all are processed and written, testing
        the full pipeline for deadlocks and data loss.
        """
        num_jobs = 100
        processor = AdvancedMetricsProcessor(output_dir=str(tmp_path), num_cpu_workers=4)
        processor.start()
        
        mask = np.zeros((32, 32), dtype=np.uint8)
        for i in range(num_jobs):
            processor.dispatch_image_job(f"img_{i}", mask, mask)
        
        processor.shutdown()

        output_file = tmp_path / "advanced_metrics.jsonl"
        assert output_file.exists()
        with open(output_file, 'r') as f:
            lines = f.readlines()
        assert len(lines) == num_jobs
        assert len({json.loads(line)['image_id'] for line in lines}) == num_jobs

    @pytest.mark.slow
    @pytest.mark.optdeps
    def test_memory_usage_under_load(self, tmp_path: Path):
        """Test that memory usage remains bounded under high load."""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil is required for memory usage tests.")

        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        processor = AdvancedMetricsProcessor(output_dir=str(tmp_path), num_cpu_workers=2)
        processor.start()

        try:
            num_jobs = 50
            large_mask = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
            for i in range(num_jobs):
                processor.dispatch_image_job(f"mem_test_{i}", large_mask, large_mask)
        finally:
            processor.shutdown()
        
        final_memory = process.memory_info().rss
        memory_growth_mb = (final_memory - initial_memory) / (1024 ** 2)
        
        # Allow for reasonable memory fluctuation and overhead, but not unbounded growth.
        assert memory_growth_mb < 200, f"Excessive memory growth detected: {memory_growth_mb:.1f} MB"

    # --- Configuration and Robustness Tests ---

    def test_task_gating_writes_only_enabled_metrics(self, tmp_path: Path):
        """
        Verify that the output record only contains keys for the tasks
        explicitly enabled in the constructor.
        """
        enabled_tasks = ["ARI", "ASSD"]
        processor = AdvancedMetricsProcessor(output_dir=str(tmp_path), num_cpu_workers=1, enabled_tasks=enabled_tasks)
        processor.start()
        
        mask = np.array([[1, 0], [0, 2]], dtype=np.uint8)
        processor.dispatch_image_job("img_gated", mask, mask)
        processor.shutdown()

        with open(tmp_path / "advanced_metrics.jsonl", 'r') as f:
            record = json.loads(f.readline())

        assert "image_id" in record
        for task in enabled_tasks:
            assert task in record
        assert "HD95" not in record
        assert "PanopticQuality" not in record

    @pytest.mark.optdeps
    def test_missing_dependency_is_handled_gracefully(self, tmp_path: Path):
        """
        Verify that a missing optional library does not crash the worker.
        """
        with patch.dict(sys.modules, {'SimpleITK': None}):
            processor = AdvancedMetricsProcessor(output_dir=str(tmp_path), num_cpu_workers=1, enabled_tasks=["ASSD", "ARI"])
            processor.start()
            try:
                processor.dispatch_image_job("img_no_sitk", np.ones((8,8)), np.ones((8,8)))
            finally:
                processor.shutdown()

        with open(tmp_path / "advanced_metrics.jsonl", 'r') as f:
            record = json.loads(f.readline())

        assert "ARI" in record
        assert record.get("ASSD", 0.0) == 0.0

    # --- Data Handling Tests ---

    @pytest.mark.parametrize("input_type", ["numpy", "torch_cpu", "torch_gpu"])
    def test_dispatch_handles_different_input_types(self, processor: AdvancedMetricsProcessor, input_type):
        """
        Verify `dispatch_image_job` correctly handles NumPy arrays and PyTorch tensors.
        """
        mask_data = np.array([[1, 0]], dtype=np.uint8)

        if input_type == "torch_gpu":
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
            mask = torch.from_numpy(mask_data).to("cuda")
        elif input_type == "torch_cpu":
            mask = torch.from_numpy(mask_data)
        else:
            mask = mask_data

        processor.job_queue = queue.Queue()
        processor._active = True

        try:
            processor.dispatch_image_job("img_types", mask, mask)
            job = processor.job_queue.get_nowait()
        finally:
            processor._active = False
            processor.job_queue = None

        img_id, pred_np, target_np = job
        assert img_id == "img_types"
        assert isinstance(pred_np, np.ndarray) and pred_np.dtype == np.uint8
        assert isinstance(target_np, np.ndarray) and target_np.dtype == np.uint8