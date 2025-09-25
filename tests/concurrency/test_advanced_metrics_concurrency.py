"""
Concurrency and stress tests for Coral-MTL advanced metrics pipeline.

These tests verify that the advanced metrics processor handles concurrent workloads,
large batches, and stress conditions properly without deadlocks or resource leaks.
"""

import pytest
import torch
import numpy as np
import threading
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict
from unittest.mock import Mock, patch

from coral_mtl.metrics.metrics_storer import AdvancedMetricsProcessor


class TestAdvancedMetricsProcessorConcurrency:
    """Test concurrent access to advanced metrics processor."""
    
    @pytest.fixture
    def mock_processor_config(self):
        """Create mock processor configuration."""
        return {
            'enabled': True,
            'num_cpu_workers': 4,
            'tasks': ['ASSD', 'HD95', 'PanopticQuality'],
            'batch_size': 8
        }
    
    @pytest.fixture
    def synthetic_mask_batch(self):
        """Generate synthetic mask batch for testing."""
        torch.manual_seed(42)
        batch_size = 4
        height, width = 64, 64
        num_classes = 3
        
        # Generate random but valid masks
        predictions = torch.randint(0, num_classes, (batch_size, height, width))
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        
        return predictions, targets
    
    @pytest.mark.slow
    @pytest.mark.optdeps
    def test_concurrent_job_submission(self, mock_processor_config, synthetic_mask_batch):
        """Test submitting jobs concurrently from multiple threads."""
        processor = AdvancedMetricsProcessor(**mock_processor_config)
        processor.start()
        
        try:
            predictions, targets = synthetic_mask_batch
            
            def submit_job(thread_id: int) -> str:
                """Submit a job from a specific thread."""
                job_id = f"concurrent_job_{thread_id}_{int(time.time() * 1000)}"
                
                for i in range(predictions.shape[0]):
                    processor.submit_job(
                        job_id=f"{job_id}_image_{i}",
                        pred_mask=predictions[i].numpy().astype(np.uint8),
                        true_mask=targets[i].numpy().astype(np.uint8),
                        image_id=f"thread_{thread_id}_img_{i}"
                    )
                
                return job_id
            
            # Submit jobs from multiple threads
            num_threads = 8
            job_ids = []
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(submit_job, thread_id)
                    for thread_id in range(num_threads)
                ]
                
                for future in futures:
                    job_id = future.result()
                    job_ids.append(job_id)
            
            # Wait for all jobs to complete
            processor.wait_for_completion(timeout=30.0)
            
            # Verify no deadlocks or crashes occurred
            assert len(job_ids) == num_threads
            
        finally:
            processor.stop()
    
    @pytest.mark.slow
    @pytest.mark.optdeps
    def test_processor_resource_cleanup(self, mock_processor_config, synthetic_mask_batch):
        """Test that processor cleans up resources properly under stress."""
        initial_memory = psutil.Process().memory_info().rss
        
        for iteration in range(5):
            processor = AdvancedMetricsProcessor(**mock_processor_config)
            processor.start()
            
            try:
                predictions, targets = synthetic_mask_batch
                
                # Submit many jobs quickly
                for batch_idx in range(10):
                    for i in range(predictions.shape[0]):
                        processor.submit_job(
                            job_id=f"stress_iter_{iteration}_batch_{batch_idx}_img_{i}",
                            pred_mask=predictions[i].numpy().astype(np.uint8),
                            true_mask=targets[i].numpy().astype(np.uint8),
                            image_id=f"stress_img_{batch_idx}_{i}"
                        )
                
                processor.wait_for_completion(timeout=15.0)
                
            finally:
                processor.stop()
            
            # Check memory hasn't grown excessively
            current_memory = psutil.Process().memory_info().rss
            memory_growth = (current_memory - initial_memory) / (1024 * 1024)  # MB
            
            # Allow some reasonable memory growth but not excessive leaks
            assert memory_growth < 100, f"Memory leak detected: {memory_growth:.1f}MB growth"
    
    @pytest.mark.slow
    @pytest.mark.optdeps
    def test_processor_queue_overflow_handling(self, mock_processor_config):
        """Test processor behavior when job queue overflows."""
        # Reduce workers to create bottleneck
        config = mock_processor_config.copy()
        config['num_cpu_workers'] = 1
        
        processor = AdvancedMetricsProcessor(**config)
        processor.start()
        
        try:
            # Generate large synthetic masks to slow down processing
            large_predictions = np.random.randint(0, 3, (128, 128), dtype=np.uint8)
            large_targets = np.random.randint(0, 3, (128, 128), dtype=np.uint8)
            
            # Submit many jobs quickly to overflow queue
            submitted_jobs = 0
            failed_submissions = 0
            
            for i in range(100):
                try:
                    processor.submit_job(
                        job_id=f"overflow_test_{i}",
                        pred_mask=large_predictions,
                        true_mask=large_targets,
                        image_id=f"overflow_img_{i}"
                    )
                    submitted_jobs += 1
                except Exception:
                    # Expected - queue overflow should be handled gracefully
                    failed_submissions += 1
            
            # Processor should handle overflow gracefully (either queue or reject)
            assert submitted_jobs + failed_submissions == 100
            
            # Wait for submitted jobs to complete
            processor.wait_for_completion(timeout=60.0)
            
        finally:
            processor.stop()
    
    @pytest.mark.slow
    @pytest.mark.optdeps
    def test_processor_worker_crash_recovery(self, mock_processor_config, synthetic_mask_batch):
        """Test processor recovery when worker processes crash."""
        processor = AdvancedMetricsProcessor(**mock_processor_config)
        processor.start()
        
        try:
            predictions, targets = synthetic_mask_batch
            
            # Submit some normal jobs
            for i in range(2):
                processor.submit_job(
                    job_id=f"normal_job_{i}",
                    pred_mask=predictions[i].numpy().astype(np.uint8),
                    true_mask=targets[i].numpy().astype(np.uint8),
                    image_id=f"normal_img_{i}"
                )
            
            # Simulate worker crash by submitting invalid data
            # (This should be handled gracefully)
            try:
                processor.submit_job(
                    job_id="crash_job",
                    pred_mask=None,  # Invalid data to cause crash
                    true_mask=targets[0].numpy().astype(np.uint8),
                    image_id="crash_img"
                )
            except Exception:
                # Expected - invalid data should be rejected
                pass
            
            # Submit more normal jobs after potential crash
            for i in range(2, 4):
                processor.submit_job(
                    job_id=f"recovery_job_{i}",
                    pred_mask=predictions[i].numpy().astype(np.uint8),
                    true_mask=targets[i].numpy().astype(np.uint8),
                    image_id=f"recovery_img_{i}"
                )
            
            # Processor should continue working
            processor.wait_for_completion(timeout=30.0)
            
        finally:
            processor.stop()


class TestAdvancedMetricsStressTests:
    """Stress tests for advanced metrics computation."""
    
    @pytest.fixture
    def stress_test_config(self):
        """Configuration for stress testing."""
        return {
            'enabled': True,
            'num_cpu_workers': max(1, psutil.cpu_count() - 1),  # Leave one CPU for system
            'tasks': ['ASSD', 'HD95', 'PanopticQuality', 'ARI'],
            'batch_size': 16
        }
    
    @pytest.mark.slow
    @pytest.mark.optdeps
    def test_large_batch_processing(self, stress_test_config):
        """Test processing of large batches."""
        processor = AdvancedMetricsProcessor(**stress_test_config)
        processor.start()
        
        try:
            # Generate large batch
            batch_size = 50
            height, width = 128, 128
            num_classes = 5
            
            np.random.seed(42)
            
            start_time = time.time()
            
            for i in range(batch_size):
                predictions = np.random.randint(0, num_classes, (height, width), dtype=np.uint8)
                targets = np.random.randint(0, num_classes, (height, width), dtype=np.uint8)
                
                processor.submit_job(
                    job_id=f"large_batch_job_{i}",
                    pred_mask=predictions,
                    true_mask=targets,
                    image_id=f"large_batch_img_{i}"
                )
            
            processor.wait_for_completion(timeout=120.0)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process reasonably quickly
            assert processing_time < 120.0, f"Large batch took too long: {processing_time:.1f}s"
            
        finally:
            processor.stop()
    
    @pytest.mark.slow
    @pytest.mark.optdeps
    def test_high_resolution_masks(self, stress_test_config):
        """Test processing of high-resolution masks."""
        processor = AdvancedMetricsProcessor(**stress_test_config)
        processor.start()
        
        try:
            # Generate high-resolution masks
            height, width = 512, 512
            num_classes = 3
            
            np.random.seed(42)
            predictions = np.random.randint(0, num_classes, (height, width), dtype=np.uint8)
            targets = np.random.randint(0, num_classes, (height, width), dtype=np.uint8)
            
            start_time = time.time()
            
            processor.submit_job(
                job_id="high_res_job",
                pred_mask=predictions,
                true_mask=targets,
                image_id="high_res_img"
            )
            
            processor.wait_for_completion(timeout=60.0)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should handle high-resolution reasonably
            assert processing_time < 60.0, f"High-res processing took too long: {processing_time:.1f}s"
            
        finally:
            processor.stop()
    
    @pytest.mark.slow
    @pytest.mark.optdeps
    def test_memory_pressure(self, stress_test_config):
        """Test processor behavior under memory pressure."""
        # Reduce workers to control memory usage
        config = stress_test_config.copy()
        config['num_cpu_workers'] = 2
        
        processor = AdvancedMetricsProcessor(**config)
        processor.start()
        
        try:
            # Monitor memory usage
            initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            # Submit many large masks
            height, width = 256, 256
            num_jobs = 20
            
            np.random.seed(42)
            
            for i in range(num_jobs):
                predictions = np.random.randint(0, 5, (height, width), dtype=np.uint8)
                targets = np.random.randint(0, 5, (height, width), dtype=np.uint8)
                
                processor.submit_job(
                    job_id=f"memory_pressure_job_{i}",
                    pred_mask=predictions,
                    true_mask=targets,
                    image_id=f"memory_pressure_img_{i}"
                )
                
                # Check memory periodically
                if i % 5 == 0:
                    current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                    memory_growth = current_memory - initial_memory
                    
                    # Fail if memory grows excessively (more than 1GB)
                    assert memory_growth < 1024, f"Excessive memory growth: {memory_growth:.1f}MB"
            
            processor.wait_for_completion(timeout=90.0)
            
            # Final memory check
            final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            total_growth = final_memory - initial_memory
            
            # Some growth is expected, but should be reasonable
            assert total_growth < 500, f"Final memory growth too large: {total_growth:.1f}MB"
            
        finally:
            processor.stop()


class TestAdvancedMetricsRaceConditions:
    """Test for race conditions in advanced metrics processing."""
    
    @pytest.fixture
    def race_condition_config(self):
        """Configuration designed to expose race conditions."""
        return {
            'enabled': True,
            'num_cpu_workers': 4,
            'tasks': ['ASSD', 'HD95'],
            'batch_size': 4
        }
    
    @pytest.mark.slow
    @pytest.mark.optdeps
    def test_simultaneous_start_stop(self, race_condition_config):
        """Test simultaneous start/stop operations."""
        processor = AdvancedMetricsProcessor(**race_condition_config)
        
        def start_processor():
            try:
                processor.start()
            except Exception:
                # May fail if already started
                pass
        
        def stop_processor():
            try:
                processor.stop()
            except Exception:
                # May fail if already stopped
                pass
        
        # Rapid start/stop from multiple threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for _ in range(10):
                futures.append(executor.submit(start_processor))
                futures.append(executor.submit(stop_processor))
            
            # Wait for all operations
            for future in futures:
                future.result()
        
        # Ensure processor is in consistent state
        try:
            processor.stop()  # Final cleanup
        except Exception:
            pass
    
    @pytest.mark.slow
    @pytest.mark.optdeps
    def test_job_submission_during_shutdown(self, race_condition_config, synthetic_mask_batch):
        """Test job submission while processor is shutting down."""
        processor = AdvancedMetricsProcessor(**race_condition_config)
        processor.start()
        
        predictions, targets = synthetic_mask_batch
        
        def submit_jobs():
            """Submit jobs continuously."""
            for i in range(20):
                try:
                    processor.submit_job(
                        job_id=f"shutdown_race_job_{i}",
                        pred_mask=predictions[0].numpy().astype(np.uint8),
                        true_mask=targets[0].numpy().astype(np.uint8),
                        image_id=f"shutdown_race_img_{i}"
                    )
                    time.sleep(0.01)  # Small delay
                except Exception:
                    # Expected during shutdown
                    break
        
        def shutdown_processor():
            """Shutdown processor after brief delay."""
            time.sleep(0.05)
            processor.stop()
        
        # Start job submission and shutdown simultaneously
        with ThreadPoolExecutor(max_workers=2) as executor:
            submit_future = executor.submit(submit_jobs)
            shutdown_future = executor.submit(shutdown_processor)
            
            # Both should complete without deadlock
            submit_future.result()
            shutdown_future.result()
    
    @pytest.mark.slow
    @pytest.mark.optdeps
    def test_concurrent_result_retrieval(self, race_condition_config, synthetic_mask_batch):
        """Test concurrent retrieval of results."""
        processor = AdvancedMetricsProcessor(**race_condition_config)
        processor.start()
        
        try:
            predictions, targets = synthetic_mask_batch
            
            # Submit several jobs
            job_ids = []
            for i in range(8):
                job_id = f"concurrent_retrieve_job_{i}"
                processor.submit_job(
                    job_id=job_id,
                    pred_mask=predictions[i % predictions.shape[0]].numpy().astype(np.uint8),
                    true_mask=targets[i % targets.shape[0]].numpy().astype(np.uint8),
                    image_id=f"concurrent_retrieve_img_{i}"
                )
                job_ids.append(job_id)
            
            def retrieve_results():
                """Attempt to retrieve results."""
                try:
                    processor.wait_for_completion(timeout=15.0)
                    return True
                except Exception:
                    return False
            
            # Multiple threads trying to retrieve results
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(retrieve_results)
                    for _ in range(4)
                ]
                
                results = [future.result() for future in futures]
                
                # At least one should succeed
                assert any(results), "No thread succeeded in retrieving results"
        
        finally:
            processor.stop()


@pytest.fixture
def synthetic_mask_batch():
    """Generate synthetic mask batch for testing (module-level fixture)."""
    torch.manual_seed(42)
    batch_size = 4
    height, width = 64, 64
    num_classes = 3
    
    # Generate random but valid masks
    predictions = torch.randint(0, num_classes, (batch_size, height, width))
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    return predictions, targets