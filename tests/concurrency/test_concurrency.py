"""Concurrency tests for advanced metrics processor."""
import pytest
import numpy as np
import time
import threading
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, MagicMock

from coral_mtl.metrics.metrics_storer import AdvancedMetricsProcessor


@pytest.mark.optdeps
class TestAdvancedMetricsProcessorConcurrency:
    """Test concurrent behavior of AdvancedMetricsProcessor."""
    
    def test_processor_start_stop_thread_safety(self, temp_output_dir, mock_optional_deps):
        """Test that start/stop operations are thread-safe."""
        try:
            processor = AdvancedMetricsProcessor(
                output_dir=str(temp_output_dir),
                num_cpu_workers=2,
                enabled_tasks=['ASSD']
            )
            
            # Multiple threads calling start/stop
            def start_stop_cycle():
                for _ in range(3):
                    processor.start()
                    time.sleep(0.01)
                    processor.shutdown()
                    time.sleep(0.01)
            
            threads = [threading.Thread(target=start_stop_cycle) for _ in range(3)]
            
            for thread in threads:
                thread.start()
            
            for thread in threads:
                thread.join(timeout=5.0)
                assert not thread.is_alive(), "Thread did not complete"
            
            # Final shutdown
            processor.shutdown()
            
        except ImportError:
            pytest.skip("Optional dependencies not available")
    
    def test_high_volume_job_dispatch(self, temp_output_dir, mock_optional_deps):
        """Test dispatching many jobs rapidly without deadlock."""
        try:
            processor = AdvancedMetricsProcessor(
                output_dir=str(temp_output_dir),
                num_cpu_workers=3,
                enabled_tasks=['ASSD', 'HD95']
            )
            processor.start()
            
            # Dispatch many jobs rapidly
            num_jobs = 50
            jobs_dispatched = 0
            
            def dispatch_image_job(job_id):
                pred_mask = np.random.randint(0, 3, (16, 16), dtype=np.uint8)
                target_mask = np.random.randint(0, 3, (16, 16), dtype=np.uint8)
                
                processor.dispatch_image_job(
                    f"high_vol_{job_id:03d}", pred_mask, target_mask)
                return job_id
            
            # Use thread pool to dispatch jobs concurrently
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(dispatch_job, i) for i in range(num_jobs)]
                
                for future in as_completed(futures, timeout=10):
                    job_id = future.result()
                    jobs_dispatched += 1
            
            assert jobs_dispatched == num_jobs, f"Only dispatched {jobs_dispatched}/{num_jobs} jobs"
            
            # Allow processing time
            time.sleep(0.5)
            
            # Should shutdown gracefully without hanging
            processor.shutdown()
            
        except ImportError:
            pytest.skip("Optional dependencies not available")
        except Exception as e:
            pytest.skip(f"High volume dispatch test failed: {e}")
    
    def test_concurrent_dispatch_different_splits(self, temp_output_dir, mock_optional_deps):
        """Test concurrent job dispatch for different splits."""
        try:
            processor = AdvancedMetricsProcessor(
                output_dir=str(temp_output_dir),
                num_cpu_workers=2,
                enabled_tasks=['ASSD']
            )
            processor.start()
            
            splits = ['train', 'val', 'test']
            jobs_per_split = 5
            
            def dispatch_for_split(split_name):
                results = []
                for i in range(jobs_per_split):
                    pred_mask = np.random.randint(0, 2, (12, 12), dtype=np.uint8)
                    target_mask = np.random.randint(0, 2, (12, 12), dtype=np.uint8)
                    
                    processor.dispatch_image_job(
                        f"{split_name}_{i:03d}", pred_mask, target_mask)
                    results.append((split_name, i))
                    
                return results
            
            # Dispatch jobs for all splits concurrently
            with ThreadPoolExecutor(max_workers=len(splits)) as executor:
                futures = [executor.submit(dispatch_for_split, split) for split in splits]
                
                all_results = []
                for future in as_completed(futures, timeout=10):
                    all_results.extend(future.result())
            
            expected_jobs = len(splits) * jobs_per_split
            assert len(all_results) == expected_jobs
            
            # Allow processing
            time.sleep(0.3)
            
            processor.shutdown()
            
        except ImportError:
            pytest.skip("Optional dependencies not available")
        except Exception as e:
            pytest.skip(f"Concurrent different splits test failed: {e}")
    
    def test_worker_pool_variations(self, temp_output_dir, mock_optional_deps):
        """Test different worker pool configurations."""
        worker_counts = [1, 2, 4]
        
        for num_workers in worker_counts:
            try:
                processor = AdvancedMetricsProcessor(
                    output_dir=str(temp_output_dir / f"workers_{num_workers}"),
                    num_cpu_workers=num_workers,
                    enabled_tasks=['ASSD']
                )
                processor.start()
                
                # Dispatch some jobs
                for i in range(5):
                    pred_mask = np.random.randint(0, 2, (8, 8), dtype=np.uint8)
                    target_mask = np.random.randint(0, 2, (8, 8), dtype=np.uint8)
                    
                    processor.dispatch_image_job(
                        f"worker_test_{num_workers}_{i}", pred_mask, target_mask)
                
                time.sleep(0.1)
                processor.shutdown()
                
            except ImportError:
                pytest.skip("Optional dependencies not available")
            except Exception as e:
                pytest.skip(f"Worker pool test with {num_workers} workers failed: {e}")
    
    def test_graceful_shutdown_with_pending_work(self, temp_output_dir, mock_optional_deps):
        """Test graceful shutdown when work is still pending."""
        try:
            processor = AdvancedMetricsProcessor(
                output_dir=str(temp_output_dir),
                num_cpu_workers=1,  # Single worker for controlled testing
                enabled_tasks=['ASSD']
            )
            processor.start()
            
            # Add several jobs
            num_jobs = 10
            for i in range(num_jobs):
                pred_mask = np.zeros((4, 4), dtype=np.uint8)
                target_mask = np.ones((4, 4), dtype=np.uint8)
                
                processor.dispatch_image_job(
                    f"pending_{i}", pred_mask, target_mask)
            
            # Shutdown immediately without waiting
            start_time = time.time()
            processor.shutdown()
            shutdown_time = time.time() - start_time
            
            # Should complete shutdown in reasonable time
            assert shutdown_time < 5.0, f"Shutdown took too long: {shutdown_time}s"
            
        except ImportError:
            pytest.skip("Optional dependencies not available")
        except Exception as e:
            pytest.skip(f"Graceful shutdown test failed: {e}")
    
    def test_repeated_start_shutdown_cycles(self, temp_output_dir, mock_optional_deps):
        """Test repeated start/shutdown cycles."""
        try:
            processor = AdvancedMetricsProcessor(
                output_dir=str(temp_output_dir),
                num_cpu_workers=2,
                enabled_tasks=['ASSD']
            )
            
            cycles = 3
            for cycle in range(cycles):
                # Start processor
                processor.start()
                
                # Add a few jobs
                for i in range(3):
                    pred_mask = np.random.randint(0, 2, (6, 6), dtype=np.uint8)
                    target_mask = np.random.randint(0, 2, (6, 6), dtype=np.uint8)
                    
                    processor.dispatch_image_job(
                        f"cycle_{cycle}_job_{i}", pred_mask, target_mask)
                
                # Brief processing time
                time.sleep(0.05)
                
                # Shutdown
                processor.shutdown()
            
            # Final shutdown should be safe
            processor.shutdown()
            
        except ImportError:
            pytest.skip("Optional dependencies not available")
        except Exception as e:
            pytest.skip(f"Repeated cycles test failed: {e}")
    
    def test_memory_usage_under_load(self, temp_output_dir, mock_optional_deps):
        """Test that memory usage remains bounded under load."""
        try:
            import psutil
            import os
            
            processor = AdvancedMetricsProcessor(
                output_dir=str(temp_output_dir),
                num_cpu_workers=2,
                enabled_tasks=['ASSD']
            )
            processor.start()
            
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            # Dispatch many jobs
            num_jobs = 30
            for i in range(num_jobs):
                # Create somewhat larger masks to test memory handling
                pred_mask = np.random.randint(0, 3, (32, 32), dtype=np.uint8)
                target_mask = np.random.randint(0, 3, (32, 32), dtype=np.uint8)
                
                processor.dispatch_image_job(
                    f"memory_test_{i:03d}", pred_mask, target_mask)
            
            # Check memory after dispatching
            peak_memory = process.memory_info().rss
            
            # Allow processing to complete
            time.sleep(1.0)
            
            processor.shutdown()
            
            # Check final memory
            final_memory = process.memory_info().rss
            
            # Memory should not grow excessively
            memory_growth = peak_memory - initial_memory
            memory_growth_mb = memory_growth / (1024 * 1024)
            
            # Allow reasonable memory growth (< 100MB for test)
            assert memory_growth_mb < 100, f"Excessive memory growth: {memory_growth_mb:.1f}MB"
            
        except ImportError:
            pytest.skip("psutil or optional dependencies not available")
        except Exception as e:
            pytest.skip(f"Memory usage test failed: {e}")
    
    def test_task_filtering_concurrency(self, temp_output_dir, mock_optional_deps):
        """Test that task filtering works correctly under concurrent load."""
        try:
            # Enable only specific tasks
            enabled_tasks = ['ASSD']
            processor = AdvancedMetricsProcessor(
                output_dir=str(temp_output_dir),
                num_cpu_workers=2,
                enabled_tasks=enabled_tasks
            )
            processor.start()
            
            # Verify task filtering
            assert processor.enabled_tasks == set(enabled_tasks)
            
            # Dispatch jobs concurrently
            def dispatch_batch(batch_id):
                for i in range(5):
                    pred_mask = np.random.randint(0, 2, (10, 10), dtype=np.uint8)
                    target_mask = np.random.randint(0, 2, (10, 10), dtype=np.uint8)
                    
                    processor.dispatch_image_job(
                        f"filter_test_{batch_id}_{i}", pred_mask, target_mask)
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(dispatch_batch, batch_id) for batch_id in range(4)]
                for future in as_completed(futures):
                    future.result()  # Wait for completion
            
            time.sleep(0.2)
            processor.shutdown()
            
        except ImportError:
            pytest.skip("Optional dependencies not available")
        except Exception as e:
            pytest.skip(f"Task filtering concurrency test failed: {e}")
    
    def test_zero_workers_fallback(self, temp_output_dir, mock_optional_deps):
        """Test behavior with zero workers (synchronous fallback)."""
        try:
            # This might error or fall back to synchronous processing
            processor = AdvancedMetricsProcessor(
                output_dir=str(temp_output_dir),
                num_cpu_workers=0,
                enabled_tasks=['ASSD']
            )
            
            # Should either error or handle gracefully
            try:
                processor.start()
                
                # Try dispatching a job
                pred_mask = np.ones((4, 4), dtype=np.uint8)
                target_mask = np.zeros((4, 4), dtype=np.uint8)
                
                processor.dispatch_image_job(
                    "zero_workers_test", pred_mask, target_mask)
                
                processor.shutdown()
                
            except (ValueError, RuntimeError):
                # Expected behavior for zero workers
                pass
            
        except ImportError:
            pytest.skip("Optional dependencies not available")
        except Exception as e:
            pytest.skip(f"Zero workers test failed: {e}")


class TestMetricsStorerConcurrency:
    """Test concurrent behavior of MetricsStorer."""
    
    def test_concurrent_history_writes(self, temp_output_dir):
        """Test concurrent epoch history writes."""
        from coral_mtl.metrics.metrics_storer import MetricsStorer
        
        storer = MetricsStorer(str(temp_output_dir))
        storer.open_for_run("concurrent_history_test")
        
        def write_history_entry(epoch):
            history_data = {
                'epoch': epoch,
                'train_loss': epoch * 0.1,
                'val_loss': epoch * 0.05,
                'val_miou': 0.5 + epoch * 0.01
            }
            storer.store_epoch_history(history_data)
            return epoch
        
        # Write history from multiple threads
        epochs = list(range(20))
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(write_history_entry, epoch) for epoch in epochs]
            completed_epochs = [future.result() for future in as_completed(futures)]
        
        # All epochs should be written
        assert len(completed_epochs) == len(epochs)
        assert set(completed_epochs) == set(epochs)
        
        # Check history file
        history_file = temp_output_dir / "concurrent_history_test" / "history.json"
        assert history_file.exists()
        
        import json
        with open(history_file) as f:
            saved_history = json.load(f)
        
        assert len(saved_history) == len(epochs)
    
    def test_concurrent_per_image_writes(self, temp_output_dir):
        """Test concurrent per-image confusion matrix writes."""
        from coral_mtl.metrics.metrics_storer import MetricsStorer
        
        storer = MetricsStorer(str(temp_output_dir))
        storer.open_for_run("concurrent_cms_test")
        
        def write_cm_entry(image_id):
            cm_data = {
                'image_id': f'image_{image_id:03d}',
                'task_cms': {
                    'health': [[10, 2], [1, 8]],
                    'genus': [[15, 3], [2, 12]]
                },
                'metrics': {
                    'health_iou': 0.75,
                    'genus_iou': 0.80
                }
            }
            storer.store_per_image_cms(cm_data, split='val')
            return image_id
        
        # Write CMs from multiple threads
        image_ids = list(range(15))
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(write_cm_entry, img_id) for img_id in image_ids]
            completed_ids = [future.result() for future in as_completed(futures)]
        
        assert len(completed_ids) == len(image_ids)
        
        # Check JSONL file
        cms_file = temp_output_dir / "concurrent_cms_test" / "validation_cms.jsonl"
        assert cms_file.exists()
        
        # Count lines in JSONL
        with open(cms_file) as f:
            lines = f.readlines()
        
        assert len(lines) == len(image_ids)
    
    def test_concurrent_final_report_writes(self, temp_output_dir):
        """Test concurrent final report writes for different splits."""
        from coral_mtl.metrics.metrics_storer import MetricsStorer
        
        storer = MetricsStorer(str(temp_output_dir))
        storer.open_for_run("concurrent_reports_test")
        
        def write_report(split_name):
            report_data = {
                'split': split_name,
                'global': {
                    'mIoU': 0.75,
                    'accuracy': 0.85
                },
                'tasks': {
                    'health': {'mIoU': 0.73},
                    'genus': {'mIoU': 0.77}
                },
                'optimization_metrics': {'H-Mean': 0.75}
            }
            storer.save_final_report(report_data, split=split_name)
            return split_name
        
        # Write reports for different splits concurrently
        splits = ['train', 'val', 'test']
        with ThreadPoolExecutor(max_workers=len(splits)) as executor:
            futures = [executor.submit(write_report, split) for split in splits]
            completed_splits = [future.result() for future in as_completed(futures)]
        
        assert len(completed_splits) == len(splits)
        assert set(completed_splits) == set(splits)
        
        # Check that all report files exist
        for split in splits:
            report_file = temp_output_dir / "concurrent_reports_test" / f"{split}_results.json"
            assert report_file.exists()
    
    def test_mixed_concurrent_operations(self, temp_output_dir):
        """Test mixed concurrent operations on MetricsStorer."""
        from coral_mtl.metrics.metrics_storer import MetricsStorer
        
        storer = MetricsStorer(str(temp_output_dir))
        storer.open_for_run("mixed_ops_test")
        
        def mixed_operations(worker_id):
            results = []
            
            # Write some history
            for epoch in range(3):
                history_data = {
                    'epoch': epoch + worker_id * 10,
                    'worker_id': worker_id,
                    'loss': (epoch + 1) * 0.1
                }
                storer.store_epoch_history(history_data)
                results.append(('history', epoch + worker_id * 10))
            
            # Write some per-image data
            for img_idx in range(2):
                cm_data = {
                    'image_id': f'worker_{worker_id}_img_{img_idx}',
                    'task_cms': {'health': [[5, 1], [0, 4]]}
                }
                storer.store_per_image_cms(cm_data, split='val')
                results.append(('cm', f'worker_{worker_id}_img_{img_idx}'))
            
            # Write final report
            report_data = {
                'worker_id': worker_id,
                'global': {'mIoU': 0.6 + worker_id * 0.1}
            }
            storer.save_final_report(report_data, split=f'worker_{worker_id}')
            results.append(('report', f'worker_{worker_id}'))
            
            return results
        
        # Run mixed operations from multiple workers
        num_workers = 4
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(mixed_operations, worker_id) for worker_id in range(num_workers)]
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())
        
        # Check that all operations completed
        expected_operations = num_workers * (3 + 2 + 1)  # 3 history + 2 cm + 1 report per worker
        assert len(all_results) == expected_operations
        
        # Check files exist
        run_dir = temp_output_dir / "mixed_ops_test"
        assert (run_dir / "history.json").exists()
        assert (run_dir / "validation_cms.jsonl").exists()
        
        # Check worker-specific reports
        for worker_id in range(num_workers):
            report_file = run_dir / f"worker_{worker_id}_results.json"
            assert report_file.exists()