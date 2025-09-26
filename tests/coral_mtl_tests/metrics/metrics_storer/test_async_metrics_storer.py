# Edit file: tests/coral_mtl/metrics/metrics_storer/test_async_metrics_storer.py
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import numpy as np
import pytest

from coral_mtl.metrics.metrics_storer import AsyncMetricsStorer


class TestAsyncMetricsStorer:
    """
    Extensive test cases for the thread-based, non-blocking AsyncMetricsStorer.
    These tests verify its lifecycle, asynchronous behavior, data integrity,
    and robustness under concurrent load.
    """

    @pytest.fixture
    def storer(self, tmp_path: Path) -> Generator[AsyncMetricsStorer, None, None]:
        """Produces an initialized AsyncMetricsStorer and ensures its shutdown."""
        storer_instance = AsyncMetricsStorer(output_dir=str(tmp_path))
        yield storer_instance
        storer_instance.shutdown()

    # --- Lifecycle and State Management Tests ---

    def test_start_worker_is_idempotent(self, storer: AsyncMetricsStorer) -> None:
        """Verifies that the background worker thread is started only once."""
        storer._start_worker_if_needed()
        first_thread = storer._worker_thread
        assert first_thread is not None

        storer._start_worker_if_needed()
        assert storer._worker_thread is first_thread

    def test_shutdown_is_safe_and_idempotent(self, storer: AsyncMetricsStorer) -> None:
        """Verifies that shutdown can be called multiple times without error, even if not started."""
        storer.shutdown()
        storer.open_for_run()
        storer.shutdown()
        storer.shutdown()

    # --- Asynchronous Behavior and Concurrency Tests ---

    def test_store_operations_are_non_blocking(self, storer: AsyncMetricsStorer) -> None:
        """
        Proves the core async contract: I/O operations return immediately,
        not waiting for the slow disk write to complete.
        """
        with patch.object(storer, "_process_storage_task") as mock_process:
            mock_process.side_effect = lambda *args, **kwargs: time.sleep(0.1)

            storer.open_for_run()

            start_time = time.time()
            for i in range(5):
                storer.store_per_image_cms(f"img_{i}", {"task": np.array([1])})
            duration = time.time() - start_time

            assert duration < 0.1, "Store operations should be non-blocking and return instantly."
            storer.wait_for_completion()
            assert mock_process.call_count == 6  # 1 for open_file, 5 for store_per_image

    def test_high_volume_mixed_operations(self, storer: AsyncMetricsStorer, tmp_path: Path) -> None:
        """
        Stress test: Concurrently queue a large number of mixed operations and verify
        all are eventually processed correctly.
        """
        num_threads = 8
        jobs_per_thread = 20
        total_cms_jobs = num_threads * jobs_per_thread

        def worker_task(thread_id: int) -> None:
            is_testing = thread_id % 2 == 0
            storer.open_for_run(is_testing=is_testing)
            for i in range(jobs_per_thread):
                image_id = f"img_thread{thread_id}_{i}"
                cm_data = {"task": np.array([[i]])}
                storer.store_per_image_cms(image_id, cm_data, is_testing=is_testing)
            storer.save_final_report({"final_metric": thread_id}, f"report_{thread_id}.json")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_threads)]
            for future in futures:
                future.result()

        storer.shutdown()

        validation_file = tmp_path / "validation_cms.jsonl"
        test_file = tmp_path / "test_cms.jsonl"

        total_lines = 0
        if validation_file.exists():
            with validation_file.open("r") as f:
                total_lines += len(f.readlines())
        if test_file.exists():
            with test_file.open("r") as f:
                total_lines += len(f.readlines())

        assert total_lines == total_cms_jobs, "Not all per-image records were written."

        for i in range(num_threads):
            report_file = tmp_path / f"report_{i}.json"
            assert report_file.exists(), f"Final report for thread {i} was not written."
            with report_file.open("r") as f:
                data = json.load(f)
                assert data["final_metric"] == i

    @pytest.mark.optdeps
    def test_store_per_image_with_metrics_in_background(self, storer: AsyncMetricsStorer, tmp_path: Path) -> None:
        """
        Verifies that scientific metrics (mIoU, BIoU) are correctly computed and stored
        by the background worker thread.
        """
        storer.open_for_run(is_testing=False)

        pred_mask = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        target_mask = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]])
        cm = {"task_a": np.array([[4, 1], [1, 3]])}

        storer.store_per_image_cms_with_metrics(
            image_id="img_metrics_01",
            confusion_matrices=cm,
            predicted_masks={"task_a": pred_mask},
            target_masks={"task_a": target_mask},
            num_classes_per_task={"task_a": 2},
        )
        storer.wait_for_completion()

        jsonl_file = tmp_path / "validation_cms.jsonl"
        assert jsonl_file.exists(), "Expected validation_cms.jsonl to be written."

        with jsonl_file.open("r") as f:
            record = json.loads(f.readline())

        metrics = record["per_image_metrics"]["task_a"]
        expected_miou = ((4 / 6) + (3 / 5)) / 2
        assert metrics["mIoU"] == pytest.approx(expected_miou)
        assert metrics["BIoU"] > 0

    def test_validation_and_test_streams_are_isolated(self, storer: AsyncMetricsStorer, tmp_path: Path) -> None:
        """
        Ensure that validation and test writes are routed to their respective files
        even when interleaved in a single run.
        """
        storer.open_for_run(is_testing=False)
        cm = {"task": np.array([[1, 0], [0, 1]], dtype=np.int32)}
        storer.store_per_image_cms("img_val", cm, is_testing=False, epoch=1)

        storer.open_for_run(is_testing=True)
        storer.store_per_image_cms("img_test", cm, is_testing=True, epoch=2)

        storer.wait_for_completion()
        storer.shutdown()

        with (tmp_path / "validation_cms.jsonl").open("r") as f:
            val_record = json.loads(f.readline())
        with (tmp_path / "test_cms.jsonl").open("r") as f:
            test_record = json.loads(f.readline())

        assert val_record["image_id"] == "img_val"
        assert val_record["epoch"] == 1
        assert test_record["image_id"] == "img_test"
        assert test_record["epoch"] == 2

    def test_save_final_report_flushes_on_shutdown(self, storer: AsyncMetricsStorer, tmp_path: Path) -> None:
        """Verify that deferred report writes are flushed when shutdown is invoked."""
        payload = {"accuracy": 0.91, "loss": 0.03}
        storer.save_final_report(payload, "final_report.json")
        storer.shutdown()

        report_path = tmp_path / "final_report.json"
        assert report_path.exists()
        with report_path.open("r") as f:
            persisted = json.load(f)
        assert persisted == payload

    def test_store_per_image_records_epoch_and_metadata(self, storer: AsyncMetricsStorer, tmp_path: Path) -> None:
        """Per-image writes should carry epoch information and mask metadata when provided."""
        storer.open_for_run()
        cm = {"task": np.array([[3, 1], [0, 2]], dtype=np.int32)}
        pred_mask = {"task": np.array([[1, 1], [0, 0]], dtype=np.uint8)}
        storer.store_per_image_cms(
            image_id="img_epoch",
            confusion_matrices=cm,
            predicted_masks=pred_mask,
            epoch=7,
        )
        storer.wait_for_completion()

        jsonl_file = tmp_path / "validation_cms.jsonl"
        assert jsonl_file.exists()
        with jsonl_file.open("r") as f:
            record = json.loads(f.readline())

        assert record["epoch"] == 7
        per_metrics = record["per_image_metrics"]["task"]
        assert per_metrics["mask_shape"] == [2, 2]
        assert per_metrics["num_pixels"] == 4
        assert set(per_metrics["unique_predictions"]) == {0, 1}

    def test_store_loss_diagnostics_async(self, storer: AsyncMetricsStorer, tmp_path: Path) -> None:
        """Loss diagnostics should be queued and written to the dedicated JSONL file."""
        storer.open_for_run(is_testing=False)
        storer.store_loss_diagnostics(
            step=42,
            epoch=3,
            diagnostics={
                "strategy_type": "IMGradStrategy",
                "task_weights": {"genus": 0.5, "health": 0.5},
                "gradient_update_norm": 1.23,
            },
        )
        storer.wait_for_completion()
        storer.shutdown()

        diag_path = tmp_path / "loss_diagnostics.jsonl"
        assert diag_path.exists()
        content = diag_path.read_text().strip().splitlines()
        assert len(content) == 1
        record = json.loads(content[0])
        assert record["step"] == 42
        assert record["epoch"] == 3
        assert record["task_weights"]["genus"] == pytest.approx(0.5)

    def test_lazy_open_without_explicit_open_for_run(self, tmp_path: Path) -> None:
        """
        Regression test: Even if open_for_run() is never called, the async storer
        must lazily open output files upon first write and persist records.
        """
        storer = AsyncMetricsStorer(output_dir=str(tmp_path))
        try:
            cm = {"task": np.array([[1, 2], [3, 4]], dtype=np.int64)}
            # Write to validation without prior open_for_run
            storer.store_per_image_cms("img_val_lazy", cm, is_testing=False, epoch=11)
            # Write to test without prior open_for_run
            storer.store_per_image_cms("img_test_lazy", cm, is_testing=True, epoch=22)
            storer.wait_for_completion()

            val_path = tmp_path / "validation_cms.jsonl"
            test_path = tmp_path / "test_cms.jsonl"

            assert val_path.exists(), "validation_cms.jsonl should be created lazily."
            assert test_path.exists(), "test_cms.jsonl should be created lazily."

            val_lines = val_path.read_text().strip().splitlines()
            test_lines = test_path.read_text().strip().splitlines()
            assert len(val_lines) >= 1
            assert len(test_lines) >= 1

            vrec = json.loads(val_lines[-1])
            trec = json.loads(test_lines[-1])
            assert vrec["image_id"] == "img_val_lazy"
            assert vrec["epoch"] == 11
            assert trec["image_id"] == "img_test_lazy"
            assert trec["epoch"] == 22
        finally:
            storer.shutdown()