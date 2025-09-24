"""
Scientific correctness tests using synthetic data for Coral-MTL project.

These tests verify that losses and models behave correctly with known ground truth data.
Uses synthetic data with controlled properties to validate mathematical correctness.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Tuple

from coral_mtl import ExperimentFactory
from coral_mtl.utils.task_splitter import MTLTaskSplitter


class TestMetricsScientificCorrectness:
    """Test basic tensor operations and validation logic with synthetic data."""
    
    @pytest.mark.slow
    def test_perfect_predictions_properties(self):
        """Test that perfect predictions have expected tensor properties."""
        torch.manual_seed(42)
        batch_size, height, width = 2, 32, 32
        num_classes = 3
        
        # Create perfect predictions
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        
        # Perfect logits (high confidence on correct class)
        logits = torch.ones(batch_size, num_classes, height, width) * -10.0
        for b in range(batch_size):
            for c in range(num_classes):
                mask = targets[b] == c
                logits[b, c][mask] = 10.0
        
        # Test predictions match targets
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == targets).float().mean()
        
        assert accuracy.item() == 1.0, f"Perfect predictions should have 100% accuracy, got {accuracy.item():.4f}"
    
    @pytest.mark.slow
    def test_worst_predictions_properties(self):
        """Test that worst predictions have expected properties."""
        torch.manual_seed(42)
        batch_size, height, width = 2, 32, 32
        num_classes = 3
        
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        
        # Worst logits (high confidence on wrong classes only)
        logits = torch.ones(batch_size, num_classes, height, width) * 10.0
        for b in range(batch_size):
            for c in range(num_classes):
                mask = targets[b] == c
                logits[b, c][mask] = -10.0  # Low confidence for correct class
        
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == targets).float().mean()
        
        # Worst predictions should have very low accuracy
        assert accuracy.item() < 0.5, f"Worst predictions should have low accuracy, got {accuracy.item():.4f}"
    
    @pytest.mark.slow
    def test_cross_entropy_loss_bounds(self):
        """Test that cross-entropy loss behaves within expected bounds."""
        torch.manual_seed(42)
        batch_size, num_classes = 4, 5
        
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # Perfect predictions (very confident)
        perfect_logits = torch.ones(batch_size, num_classes) * -10.0
        for i, target in enumerate(targets):
            perfect_logits[i, target] = 10.0
        
        # Random predictions
        random_logits = torch.randn(batch_size, num_classes) * 0.1
        
        # Worst predictions (confident on wrong class)
        worst_logits = torch.ones(batch_size, num_classes) * 10.0
        for i, target in enumerate(targets):
            worst_logits[i, target] = -10.0
        
        # Calculate losses
        loss_fn = torch.nn.CrossEntropyLoss()
        perfect_loss = loss_fn(perfect_logits, targets)
        random_loss = loss_fn(random_logits, targets)
        worst_loss = loss_fn(worst_logits, targets)
        
        # Perfect < Random < Worst (generally)
        assert perfect_loss < random_loss, f"Perfect loss {perfect_loss:.4f} should be < random loss {random_loss:.4f}"
        assert random_loss < worst_loss, f"Random loss {random_loss:.4f} should be < worst loss {worst_loss:.4f}"
        
        # Perfect predictions should have very low loss
        assert perfect_loss < 0.01, f"Perfect predictions should have very low loss, got {perfect_loss:.4f}"


class TestLossScientificCorrectness:
    """Test loss functions with known synthetic ground truth."""
    
    @pytest.fixture
    def experiment_factory(self, tmp_path):
        """Create experiment factory with minimal config."""
        config_path = tmp_path / "test_config.yaml"
        config_content = """
model:
  type: "coral_mtl"
  backbone: "segformer-b0"
  
dataset:
  name: "synthetic"
  root: "."
  
loss:
  type: "CoralMTLLoss"
  
optimization_metrics:
  H-Mean: ["genus.mIoU", "health.mIoU"]
  
trainer:
  max_epochs: 1
  device: "cpu"
"""
        config_path.write_text(config_content)
        
        factory = ExperimentFactory(str(config_path))
        return factory
    
    @pytest.mark.slow
    def test_cross_entropy_loss_synthetic_perfect(self, experiment_factory):
        """Test that perfect predictions yield near-zero cross-entropy loss."""
        torch.manual_seed(42)
        
        # Create synthetic perfect predictions
        batch_size, height, width = 2, 16, 16
        num_classes = 3
        
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        
        # Perfect logits (high confidence on correct class)
        logits = torch.ones(batch_size, num_classes, height, width) * -10.0
        for b in range(batch_size):
            for c in range(num_classes):
                mask = targets[b] == c
                logits[b, c][mask] = 10.0
        
        loss_fn = experiment_factory.get_loss_function()
        
        # Test with single task (genus)
        predictions = {"genus": logits}
        ground_truth = {"genus": targets}
        
        loss_dict = loss_fn(predictions, ground_truth)
        
        # Perfect predictions should yield very low loss
        total_loss = loss_dict["loss"]
        assert total_loss < 0.01, f"Perfect predictions should yield low loss, got {total_loss}"
    
    @pytest.mark.slow
    def test_cross_entropy_loss_synthetic_worst(self, experiment_factory):
        """Test that worst predictions yield high cross-entropy loss."""
        torch.manual_seed(42)
        
        batch_size, height, width = 2, 16, 16
        num_classes = 3
        
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        
        # Worst logits (high confidence on wrong classes)
        logits = torch.ones(batch_size, num_classes, height, width) * 10.0
        for b in range(batch_size):
            for c in range(num_classes):
                mask = targets[b] == c
                logits[b, c][mask] = -10.0  # Low confidence for correct class
        
        loss_fn = experiment_factory.get_loss_function()
        
        predictions = {"genus": logits}
        ground_truth = {"genus": targets}
        
        loss_dict = loss_fn(predictions, ground_truth)
        
        # Worst predictions should yield high loss
        total_loss = loss_dict["loss"]
        assert total_loss > 2.0, f"Worst predictions should yield high loss, got {total_loss}"


class TestModelOutputScientificCorrectness:
    """Test model outputs with synthetic data."""
    
    @pytest.fixture
    def minimal_coral_mtl_model(self, tmp_path):
        """Create minimal CoralMTL model for testing."""
        config_path = tmp_path / "test_config.yaml"
        config_content = """
model:
  type: "coral_mtl"
  backbone: "segformer-b0"
  
dataset:
  root: "."
  
trainer:
  device: "cpu"
"""
        config_path.write_text(config_content)
        
        factory = ExperimentFactory(str(config_path))
        model = factory.get_model()
        model.eval()
        return model
    
    @pytest.mark.slow
    def test_model_output_shape_consistency(self, minimal_coral_mtl_model):
        """Test that model outputs have consistent shapes across different input sizes."""
        model = minimal_coral_mtl_model
        
        test_sizes = [(1, 3, 128, 128), (2, 3, 64, 64), (1, 3, 256, 256)]
        
        for batch_size, channels, height, width in test_sizes:
            torch.manual_seed(42)
            x = torch.randn(batch_size, channels, height, width)
            
            with torch.no_grad():
                outputs = model(x)
            
            # Check that outputs are dict with expected tasks
            assert isinstance(outputs, dict), "Model should return dict of task outputs"
            
            for task_name, task_output in outputs.items():
                # Check output shape matches input spatial dimensions
                assert task_output.shape[0] == batch_size, \
                    f"Batch size mismatch for task {task_name}"
                assert task_output.shape[2] == height, \
                    f"Height mismatch for task {task_name}"
                assert task_output.shape[3] == width, \
                    f"Width mismatch for task {task_name}"
                
                # Check that outputs are valid logits (finite values)
                assert torch.isfinite(task_output).all(), \
                    f"Model outputs should be finite for task {task_name}"
    
    @pytest.mark.slow
    def test_model_deterministic_output(self, minimal_coral_mtl_model):
        """Test that model produces deterministic outputs with same seed."""
        model = minimal_coral_mtl_model
        
        batch_size, channels, height, width = 1, 3, 64, 64
        
        # First run
        torch.manual_seed(42)
        x1 = torch.randn(batch_size, channels, height, width)
        with torch.no_grad():
            outputs1 = model(x1)
        
        # Second run with same seed
        torch.manual_seed(42)
        x2 = torch.randn(batch_size, channels, height, width)
        with torch.no_grad():
            outputs2 = model(x2)
        
        # Outputs should be identical
        for task_name in outputs1.keys():
            assert torch.allclose(outputs1[task_name], outputs2[task_name], atol=1e-6), \
                f"Deterministic outputs failed for task {task_name}"
    
    @pytest.mark.slow
    def test_model_gradient_flow(self, minimal_coral_mtl_model):
        """Test that gradients flow properly through model."""
        model = minimal_coral_mtl_model
        model.train()
        
        torch.manual_seed(42)
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        
        outputs = model(x)
        
        # Compute dummy loss and backpropagate
        total_loss = sum(output.sum() for output in outputs.values())
        total_loss.backward()
        
        # Check that input gradients exist
        assert x.grad is not None, "Input should have gradients"
        assert not torch.isnan(x.grad).any(), "Input gradients should not be NaN"
        
        # Check that model parameters have gradients
        params_with_grad = 0
        total_params = 0
        
        for param in model.parameters():
            total_params += 1
            if param.grad is not None:
                params_with_grad += 1
                assert not torch.isnan(param.grad).any(), "Parameter gradients should not be NaN"
        
        # Most parameters should have gradients
        grad_ratio = params_with_grad / total_params
        assert grad_ratio > 0.8, f"Most parameters should have gradients, got {grad_ratio:.2%}"


class TestTaskSplitterScientificCorrectness:
    """Test task splitter with synthetic hierarchical data."""
    
    @pytest.fixture
    def synthetic_task_definitions(self):
        """Create synthetic task definitions for testing."""
        return {
            "genus": {
                "ungrouped": {
                    "id2label": {0: "Acropora", 1: "Porites", 2: "Montipora"},
                    "label2id": {"Acropora": 0, "Porites": 1, "Montipora": 2}
                }
            },
            "health": {
                "ungrouped": {
                    "id2label": {0: "Healthy", 1: "Bleached", 2: "Dead"},
                    "label2id": {"Healthy": 0, "Bleached": 1, "Dead": 2}
                }
            }
        }
    
    @pytest.mark.slow
    def test_task_splitter_class_count_consistency(self, synthetic_task_definitions):
        """Test that task splitter reports consistent class counts."""
        splitter = MTLTaskSplitter(
            hierarchical_definitions=synthetic_task_definitions,
            primary_tasks=["genus", "health"],
            auxiliary_tasks=[]
        )
        
        # Check class counts match definitions
        for task_name in ["genus", "health"]:
            expected_classes = len(synthetic_task_definitions[task_name]["ungrouped"]["id2label"])
            actual_classes = splitter.get_num_classes(task_name)
            
            assert actual_classes == expected_classes, \
                f"Class count mismatch for {task_name}: expected {expected_classes}, got {actual_classes}"
    
    @pytest.mark.slow
    def test_task_splitter_synthetic_conversion(self, synthetic_task_definitions):
        """Test task splitter conversion with synthetic data."""
        splitter = MTLTaskSplitter(
            hierarchical_definitions=synthetic_task_definitions,
            primary_tasks=["genus", "health"],
            auxiliary_tasks=[]
        )
        
        # Create synthetic flattened labels
        torch.manual_seed(42)
        batch_size, height, width = 2, 16, 16
        
        # Simulate flattened labels (each pixel has genus + health combination)
        flat_labels = torch.randint(0, 9, (batch_size, height, width))  # 3x3 = 9 combinations
        
        # Convert to hierarchical
        hierarchical = splitter.flat_to_hierarchical(flat_labels)
        
        # Check that all expected tasks are present
        expected_tasks = {"genus", "health"}
        assert set(hierarchical.keys()) == expected_tasks, \
            f"Expected tasks {expected_tasks}, got {set(hierarchical.keys())}"
        
        # Check shapes are preserved
        for task_name, task_labels in hierarchical.items():
            assert task_labels.shape == (batch_size, height, width), \
                f"Shape mismatch for {task_name}"
            
            # Check label values are in valid range
            num_classes = len(synthetic_task_definitions[task_name]["ungrouped"]["id2label"])
            assert task_labels.min() >= 0, f"Labels should be non-negative for {task_name}"
            assert task_labels.max() < num_classes, f"Labels should be < {num_classes} for {task_name}"
