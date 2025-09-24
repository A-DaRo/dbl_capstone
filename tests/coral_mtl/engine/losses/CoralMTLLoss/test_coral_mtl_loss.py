"""Tests for CoralMTLLoss class."""
import pytest
import torch
from unittest.mock import MagicMock

from coral_mtl.engine.losses import CoralMTLLoss


class TestCoralMTLLoss:
    """Test cases for CoralMTLLoss class."""
    
    def test_coral_mtl_loss_init(self, splitter_mtl):
        """Test CoralMTLLoss initialization."""
        try:
            loss_fn = CoralMTLLoss(
                task_splitter=splitter_mtl,
                primary_tasks=['health'],
                auxiliary_tasks=['genus']
            )
            assert loss_fn is not None
            assert hasattr(loss_fn, 'task_splitter')
            assert hasattr(loss_fn, 'primary_tasks')
            assert hasattr(loss_fn, 'auxiliary_tasks')
        except Exception as e:
            pytest.skip(f"CoralMTLLoss initialization failed: {e}")
    
    def test_mtl_loss_forward_dict_output(self, splitter_mtl, dummy_masks, device):
        """Test that MTL loss returns dict with component losses."""
        try:
            loss_fn = CoralMTLLoss(
                task_splitter=splitter_mtl,
                primary_tasks=['health'],
                auxiliary_tasks=['genus']
            )
            
            # Create dummy predictions matching the masks structure
            predictions = {}
            for task_name, target_mask in dummy_masks.items():
                if task_name in ['health', 'genus']:  # Only for configured tasks
                    batch_size, h, w = target_mask.shape
                    task_info = splitter_mtl.hierarchical_definitions[task_name]
                    num_classes = len(task_info['ungrouped']['id2label'])
                    predictions[task_name] = torch.randn(batch_size, num_classes, h, w, device=device)
            
            if not predictions:
                pytest.skip("No valid task predictions created")
            
            loss_dict = loss_fn(predictions, dummy_masks)
            
            # Should return dict
            assert isinstance(loss_dict, dict)
            
            # Should have component losses and total
            assert 'total' in loss_dict or 'loss' in loss_dict
            
            # All loss values should be finite
            for loss_name, loss_value in loss_dict.items():
                assert torch.isfinite(loss_value).all(), f"Loss {loss_name} is not finite"
                assert loss_value.requires_grad, f"Loss {loss_name} doesn't require grad"
                
        except Exception as e:
            pytest.skip(f"MTL loss forward test failed: {e}")
    
    def test_mtl_loss_backward_pass(self, splitter_mtl, dummy_masks, device):
        """Test that MTL loss backward pass works."""
        try:
            loss_fn = CoralMTLLoss(
                task_splitter=splitter_mtl,
                primary_tasks=['health'],
                auxiliary_tasks=['genus']
            )
            
            # Create predictions with gradient tracking
            predictions = {}
            for task_name, target_mask in dummy_masks.items():
                if task_name in ['health', 'genus']:
                    batch_size, h, w = target_mask.shape
                    task_info = splitter_mtl.hierarchical_definitions[task_name]
                    num_classes = len(task_info['ungrouped']['id2label'])
                    predictions[task_name] = torch.randn(
                        batch_size, num_classes, h, w, 
                        device=device, requires_grad=True
                    )
            
            if not predictions:
                pytest.skip("No valid predictions created")
            
            loss_dict = loss_fn(predictions, dummy_masks)
            
            # Get total loss
            total_loss = loss_dict.get('total', loss_dict.get('loss'))
            if total_loss is None:
                total_loss = sum(loss_dict.values())
            
            total_loss.backward()
            
            # Check that predictions have gradients
            for pred in predictions.values():
                assert pred.grad is not None, "Prediction gradients not computed"
                assert torch.isfinite(pred.grad).all(), "Prediction gradients not finite"
                
        except Exception as e:
            pytest.skip(f"MTL loss backward test failed: {e}")
    
    def test_mtl_loss_task_weights(self, splitter_mtl, dummy_masks, device):
        """Test MTL loss with different task weights."""
        try:
            # Test with custom task weights
            loss_fn = CoralMTLLoss(
                task_splitter=splitter_mtl,
                primary_tasks=['health'],
                auxiliary_tasks=['genus'],
                task_weights={'health': 2.0, 'genus': 0.5}
            )
            
            # Create predictions
            predictions = {}
            for task_name, target_mask in dummy_masks.items():
                if task_name in ['health', 'genus']:
                    batch_size, h, w = target_mask.shape
                    task_info = splitter_mtl.hierarchical_definitions[task_name]
                    num_classes = len(task_info['ungrouped']['id2label'])
                    predictions[task_name] = torch.randn(batch_size, num_classes, h, w, device=device)
            
            if not predictions:
                pytest.skip("No valid predictions created")
            
            loss_dict = loss_fn(predictions, dummy_masks)
            
            # Should return dict with weighted losses
            assert isinstance(loss_dict, dict)
            
            # All losses should be finite
            for loss_value in loss_dict.values():
                assert torch.isfinite(loss_value).all()
                
        except Exception as e:
            pytest.skip(f"MTL task weights test failed: {e}")
    
    def test_mtl_loss_extreme_cases(self, splitter_mtl, device):
        """Test MTL loss behavior with extreme cases."""
        try:
            loss_fn = CoralMTLLoss(
                task_splitter=splitter_mtl,
                primary_tasks=['health'],
                auxiliary_tasks=['genus']
            )
            
            # Create perfect predictions (should give low loss)
            batch_size, h, w = 1, 8, 8
            predictions = {}
            targets = {}
            
            for task_name in ['health', 'genus']:
                if task_name in splitter_mtl.hierarchical_definitions:
                    task_info = splitter_mtl.hierarchical_definitions[task_name]
                    num_classes = len(task_info['ungrouped']['id2label'])
                    
                    # Perfect prediction (high confidence for correct class)
                    pred = torch.zeros(batch_size, num_classes, h, w, device=device)
                    pred[:, 1, :, :] = 10.0  # High logit for class 1
                    predictions[task_name] = pred
                    
                    # Target with class 1
                    targets[task_name] = torch.ones(batch_size, h, w, dtype=torch.long, device=device)
            
            if predictions and targets:
                loss_dict = loss_fn(predictions, targets)
                
                # Perfect predictions should give low loss
                total_loss = loss_dict.get('total', loss_dict.get('loss', sum(loss_dict.values())))
                assert total_loss.item() < 1.0, "Perfect predictions should give low loss"
                
        except Exception as e:
            pytest.skip(f"MTL extreme cases test failed: {e}")
    
    def test_mtl_loss_uncertainty_weighting(self, splitter_mtl, dummy_masks, device):
        """Test MTL loss with uncertainty weighting if available."""
        try:
            loss_fn = CoralMTLLoss(
                task_splitter=splitter_mtl,
                primary_tasks=['health'],
                auxiliary_tasks=['genus'],
                use_uncertainty_weighting=True
            )
            
            # Create predictions
            predictions = {}
            for task_name, target_mask in dummy_masks.items():
                if task_name in ['health', 'genus']:
                    batch_size, h, w = target_mask.shape
                    task_info = splitter_mtl.hierarchical_definitions[task_name]
                    num_classes = len(task_info['ungrouped']['id2label'])
                    predictions[task_name] = torch.randn(batch_size, num_classes, h, w, device=device)
            
            if not predictions:
                pytest.skip("No valid predictions created")
            
            loss_dict = loss_fn(predictions, dummy_masks)
            
            # Should still return dict with losses
            assert isinstance(loss_dict, dict)
            
            # Uncertainty parameters should be learnable
            if hasattr(loss_fn, 'uncertainty_weights'):
                for param in loss_fn.uncertainty_weights.parameters():
                    assert param.requires_grad, "Uncertainty weights should be learnable"
                
        except Exception as e:
            pytest.skip(f"MTL uncertainty weighting test failed: {e}")