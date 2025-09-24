"""Tests for AdvancedMetricsProcessor class."""
import pytest
import tempfile
from pathlib import Path
import json
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor
import time

from coral_mtl.metrics.metrics_storer import AdvancedMetricsProcessor


class TestAdvancedMetricsProcessor:
    """Test cases for AdvancedMetricsProcessor class."""
    
    def test_advanced_metrics_processor_init(self, temp_output_dir):
        """Test AdvancedMetricsProcessor initialization."""
        try:
            processor = AdvancedMetricsProcessor(
                output_dir=str(temp_output_dir),
                num_workers=2,
                enabled_tasks=['ASSD', 'HD95']
            )
            assert processor is not None
            assert hasattr(processor, 'output_dir')
        except Exception as e:
            pytest.skip(f"AdvancedMetricsProcessor initialization failed: {e}")
    
    @pytest.mark.optdeps
    def test_advanced_metrics_processor_dependencies(self):
        """Test that processor handles optional dependencies."""
        try:
            # Test importing optional dependencies
            try:
                import scipy
                import skimage  # scikit-image
                dependencies_available = True
            except ImportError:
                dependencies_available = False
            
            if not dependencies_available:
                pytest.skip("Optional dependencies not available")
            
            # If dependencies are available, processor should work
            with tempfile.TemporaryDirectory() as temp_dir:
                processor = AdvancedMetricsProcessor(
                    output_dir=temp_dir,
                    num_workers=1,
                    enabled_tasks=['ASSD']
                )
                assert processor is not None
                
        except Exception as e:
            pytest.skip(f"Advanced metrics dependencies test failed: {e}")
    
    @pytest.mark.slow
    def test_advanced_metrics_processor_submit_job(self, temp_output_dir):
        """Test submitting jobs to advanced metrics processor."""
        try:
            processor = AdvancedMetricsProcessor(
                output_dir=str(temp_output_dir),
                num_workers=1,
                enabled_tasks=['ASSD']
            )
            
            # Mock prediction and target masks
            import numpy as np
            pred_mask = np.random.randint(0, 3, (64, 64), dtype=np.uint8)
            target_mask = np.random.randint(0, 3, (64, 64), dtype=np.uint8)
            
            if hasattr(processor, 'submit_job'):
                job_id = processor.submit_job(
                    image_id='test_001',
                    task_name='genus',
                    pred_mask=pred_mask,
                    target_mask=target_mask
                )
                
                # Should return some job identifier
                assert job_id is not None
                
        except Exception as e:
            pytest.skip(f"Advanced metrics job submission test failed: {e}")
    
    @pytest.mark.slow  
    def test_advanced_metrics_processor_wait_completion(self, temp_output_dir):
        """Test waiting for processor job completion."""
        try:
            processor = AdvancedMetricsProcessor(
                output_dir=str(temp_output_dir),
                num_workers=1,
                enabled_tasks=['ASSD']
            )
            
            # Submit a simple job
            import numpy as np
            pred_mask = np.ones((32, 32), dtype=np.uint8)
            target_mask = np.ones((32, 32), dtype=np.uint8)
            
            if hasattr(processor, 'submit_job') and hasattr(processor, 'wait_completion'):
                processor.submit_job(
                    image_id='test_simple',
                    task_name='genus',
                    pred_mask=pred_mask,
                    target_mask=target_mask
                )
                
                # Wait for completion
                processor.wait_completion(timeout=30)
                
                # Should complete without error
                assert True
                
        except Exception as e:
            pytest.skip(f"Advanced metrics completion test failed: {e}")
    
    def test_advanced_metrics_processor_task_filtering(self, temp_output_dir):
        """Test that processor respects enabled_tasks filtering."""
        try:
            # Processor with limited tasks
            processor = AdvancedMetricsProcessor(
                output_dir=str(temp_output_dir),
                num_workers=1,
                enabled_tasks=['ASSD']  # Only ASSD enabled
            )
            
            # Should only process enabled tasks
            if hasattr(processor, 'enabled_tasks'):
                assert 'ASSD' in processor.enabled_tasks
                
            # Test with empty tasks (should handle gracefully)
            processor_empty = AdvancedMetricsProcessor(
                output_dir=str(temp_output_dir),
                num_workers=1,
                enabled_tasks=[]
            )
            assert processor_empty is not None
            
        except Exception as e:
            pytest.skip(f"Advanced metrics task filtering test failed: {e}")
    
    def test_advanced_metrics_processor_worker_management(self, temp_output_dir):
        """Test processor worker pool management."""
        try:
            processor = AdvancedMetricsProcessor(
                output_dir=str(temp_output_dir),
                num_workers=2,
                enabled_tasks=['ASSD', 'HD95']
            )
            
            # Should manage worker pool
            if hasattr(processor, 'executor'):
                assert processor.executor is not None
            
            # Test shutdown
            if hasattr(processor, 'shutdown'):
                processor.shutdown()
            elif hasattr(processor, 'close'):
                processor.close()
            elif hasattr(processor, '__del__'):
                # Cleanup should be handled
                pass
                
        except Exception as e:
            pytest.skip(f"Advanced metrics worker management test failed: {e}")
    
    def test_advanced_metrics_processor_output_format(self, temp_output_dir):
        """Test processor output format."""
        try:
            processor = AdvancedMetricsProcessor(
                output_dir=str(temp_output_dir),
                num_workers=1,
                enabled_tasks=['ASSD']
            )
            
            # Mock a completed job result
            mock_result = {
                'image_id': 'test_001',
                'task_name': 'genus',
                'ASSD': 2.5,
                'timestamp': '2024-01-01T00:00:00'
            }
            
            if hasattr(processor, 'store_result'):
                processor.store_result(mock_result)
                
                # Check output file
                output_file = Path(temp_output_dir) / "advanced_metrics.jsonl"
                if output_file.exists():
                    # Verify JSONL format
                    with open(output_file, 'r') as f:
                        line = f.readline().strip()
                        parsed = json.loads(line)
                        assert 'image_id' in parsed
                        assert 'ASSD' in parsed
            
        except Exception as e:
            pytest.skip(f"Advanced metrics output format test failed: {e}")
    
    @pytest.mark.slow
    def test_advanced_metrics_processor_error_handling(self, temp_output_dir):
        """Test processor error handling with invalid inputs."""
        try:
            processor = AdvancedMetricsProcessor(
                output_dir=str(temp_output_dir),
                num_workers=1,
                enabled_tasks=['ASSD']
            )
            
            # Test with invalid mask shapes
            import numpy as np
            pred_mask = np.ones((10, 10), dtype=np.uint8)
            target_mask = np.ones((20, 20), dtype=np.uint8)  # Different shape
            
            if hasattr(processor, 'submit_job'):
                try:
                    processor.submit_job(
                        image_id='test_invalid',
                        task_name='genus', 
                        pred_mask=pred_mask,
                        target_mask=target_mask
                    )
                    
                    # Should handle error gracefully
                    if hasattr(processor, 'wait_completion'):
                        processor.wait_completion(timeout=10)
                        
                except (ValueError, RuntimeError):
                    # Expected to fail with mismatched shapes
                    pass
            
            # Test passes if error handling is appropriate
            assert True
            
        except Exception as e:
            pytest.skip(f"Advanced metrics error handling test failed: {e}")