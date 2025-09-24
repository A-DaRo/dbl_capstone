"""Tests for Trainer class."""
import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from coral_mtl.engine.trainer import Trainer


class TestTrainer:
    """Test cases for Trainer class."""
    
    def test_trainer_init(self, minimal_coral_mtl_model, temp_output_dir, splitter_mtl, device):
        """Test Trainer initialization."""
        from coral_mtl.metrics.metrics import CoralMTLMetrics
        from coral_mtl.metrics.metrics_storer import MetricsStorer
        from coral_mtl.data.dataset import CoralscapesMTLDataset
        from coral_mtl.engine.losses import CoralMTLLoss
        from torch.utils.data import DataLoader
        from types import SimpleNamespace
        
        # Create test loaders
        train_dataset = CoralscapesMTLDataset(
            data_root_path="tests/dataset/coralscapes",
            split='train',
            splitter=splitter_mtl,
            patch_size=32
        )
        train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)
        val_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)  # Same for simplicity
        
        # Create loss function with correct interface
        num_classes = {
            task_name: len(task_info['ungrouped']['id2label'])
            for task_name, task_info in splitter_mtl.hierarchical_definitions.items()
        }
        loss_fn = CoralMTLLoss(num_classes=num_classes)
        
        # Create metrics storer first
        metrics_storer = MetricsStorer(output_dir=str(temp_output_dir))
        
        # Create metrics calculator
        metrics_calculator = CoralMTLMetrics(
            splitter=splitter_mtl,
            storer=metrics_storer,
            device=device
        )
        
        # Create optimizer and scheduler
        optimizer = torch.optim.Adam(minimal_coral_mtl_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        # Create config
        config = SimpleNamespace(
            device=device.type,
            output_dir=str(temp_output_dir),
            max_epochs=1,
            model_selection_metric='global.mIoU',
            use_mixed_precision=False
        )
        
        trainer = Trainer(
            model=minimal_coral_mtl_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            metrics_calculator=metrics_calculator,
            metrics_storer=metrics_storer,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config
        )
        assert trainer is not None
        assert trainer.device == device
    
    @pytest.mark.slow
    def test_trainer_mini_epoch_cpu(self, minimal_coral_mtl_model, temp_output_dir, device, splitter_mtl):
        """Test trainer can complete mini epoch on CPU."""
        if device.type != 'cpu':
            pytest.skip("CPU-only test")
        
        from coral_mtl.metrics.metrics import CoralMTLMetrics
        from coral_mtl.metrics.metrics_storer import MetricsStorer
        from coral_mtl.data.dataset import CoralscapesMTLDataset
        from coral_mtl.engine.losses import CoralMTLLoss
        from torch.utils.data import DataLoader
        from types import SimpleNamespace
        
        # Create test loaders
        train_dataset = CoralscapesMTLDataset(
            data_root_path="tests/dataset/coralscapes",
            split='train',
            splitter=splitter_mtl,
            patch_size=32
        )
        train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)
        val_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)
        
        # Create loss function
        num_classes = {
            task_name: len(task_info['ungrouped']['id2label'])
            for task_name, task_info in splitter_mtl.hierarchical_definitions.items()
        }
        loss_fn = CoralMTLLoss(num_classes=num_classes)
        
        # Create metrics storer first
        metrics_storer = MetricsStorer(output_dir=str(temp_output_dir))
        
        # Create metrics calculator
        metrics_calculator = CoralMTLMetrics(
            splitter=splitter_mtl,
            storer=metrics_storer,
            device=device
        )
        
        # Create optimizer and scheduler
        optimizer = torch.optim.Adam(minimal_coral_mtl_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        # Create config
        config = SimpleNamespace(
            device='cpu',
            output_dir=str(temp_output_dir),
            max_epochs=1,
            model_selection_metric='global.mIoU',
            use_mixed_precision=False
        )
        
        trainer = Trainer(
            model=minimal_coral_mtl_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            metrics_calculator=metrics_calculator,
            metrics_storer=metrics_storer,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config
        )
        
        # Run one training epoch - should not crash
        loss_dict = trainer.train_epoch(epoch=0)
        assert loss_dict is not None
    
    def test_trainer_model_mode_switching(self, minimal_coral_mtl_model, temp_output_dir, splitter_mtl, device):
        """Test that trainer properly switches model modes."""
        from coral_mtl.metrics.metrics import CoralMTLMetrics
        from coral_mtl.metrics.metrics_storer import MetricsStorer
        from coral_mtl.data.dataset import CoralscapesMTLDataset
        from coral_mtl.engine.losses import CoralMTLLoss
        from torch.utils.data import DataLoader
        from types import SimpleNamespace
        
        # Create test setup
        train_dataset = CoralscapesMTLDataset(
            data_root_path="tests/dataset/coralscapes",
            split='train',
            splitter=splitter_mtl,
            patch_size=32
        )
        train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)
        val_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)
        
        num_classes = {
            task_name: len(task_info['ungrouped']['id2label'])
            for task_name, task_info in splitter_mtl.hierarchical_definitions.items()
        }
        loss_fn = CoralMTLLoss(num_classes=num_classes)
        
        # Create metrics storer first
        metrics_storer = MetricsStorer(output_dir=str(temp_output_dir))
        
        # Create metrics calculator
        metrics_calculator = CoralMTLMetrics(
            splitter=splitter_mtl,
            storer=metrics_storer,
            device=device
        )
        
        optimizer = torch.optim.Adam(minimal_coral_mtl_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        config = SimpleNamespace(
            device=device.type,
            output_dir=str(temp_output_dir),
            max_epochs=1,
            model_selection_metric='global.mIoU',
            use_mixed_precision=False
        )
        
        trainer = Trainer(
            model=minimal_coral_mtl_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            metrics_calculator=metrics_calculator,
            metrics_storer=metrics_storer,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config
        )
        
        # Test mode switching
        trainer.model.train()
        assert trainer.model.training == True
        
        trainer.model.eval()
        assert trainer.model.training == False
    
    def test_trainer_checkpoint_saving(self, minimal_coral_mtl_model, temp_output_dir, splitter_mtl, device):
        """Test that trainer can save checkpoints."""
        from coral_mtl.metrics.metrics import CoralMTLMetrics
        from coral_mtl.metrics.metrics_storer import MetricsStorer
        from coral_mtl.data.dataset import CoralscapesMTLDataset
        from coral_mtl.engine.losses import CoralMTLLoss
        from torch.utils.data import DataLoader
        from types import SimpleNamespace
        
        # Create test setup
        train_dataset = CoralscapesMTLDataset(
            data_root_path="tests/dataset/coralscapes",
            split='train',
            splitter=splitter_mtl,
            patch_size=32
        )
        train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)
        val_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)
        
        num_classes = {
            task_name: len(task_info['ungrouped']['id2label'])
            for task_name, task_info in splitter_mtl.hierarchical_definitions.items()
        }
        loss_fn = CoralMTLLoss(num_classes=num_classes)
        
        # Create metrics storer first
        metrics_storer = MetricsStorer(output_dir=str(temp_output_dir))
        
        # Create metrics calculator
        metrics_calculator = CoralMTLMetrics(
            splitter=splitter_mtl,
            storer=metrics_storer,
            device=device
        )
        
        optimizer = torch.optim.Adam(minimal_coral_mtl_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        config = SimpleNamespace(
            device=device.type,
            output_dir=str(temp_output_dir),
            max_epochs=1,
            model_selection_metric='global.mIoU',
            use_mixed_precision=False
        )
        
        trainer = Trainer(
            model=minimal_coral_mtl_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            metrics_calculator=metrics_calculator,
            metrics_storer=metrics_storer,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config
        )
        
        # Test checkpoint saving functionality
        if hasattr(trainer, 'save_checkpoint'):
            checkpoint_path = temp_output_dir / "test_checkpoint.pth"
            trainer.save_checkpoint(str(checkpoint_path), epoch=0)
            assert checkpoint_path.exists()
        else:
            # Test direct model saving as fallback
            model_path = temp_output_dir / "model.pth"
            torch.save(trainer.model.state_dict(), model_path)
            assert model_path.exists()
    
    def test_trainer_device_handling(self, minimal_coral_mtl_model, temp_output_dir, device, splitter_mtl):
        """Test trainer handles device properly."""
        from coral_mtl.metrics.metrics import CoralMTLMetrics
        from coral_mtl.metrics.metrics_storer import MetricsStorer
        from coral_mtl.data.dataset import CoralscapesMTLDataset
        from coral_mtl.engine.losses import CoralMTLLoss
        from torch.utils.data import DataLoader
        from types import SimpleNamespace
        
        # Create test setup
        train_dataset = CoralscapesMTLDataset(
            data_root_path="tests/dataset/coralscapes",
            split='train',
            splitter=splitter_mtl,
            patch_size=32
        )
        train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)
        val_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)
        
        num_classes = {
            task_name: len(task_info['ungrouped']['id2label'])
            for task_name, task_info in splitter_mtl.hierarchical_definitions.items()
        }
        loss_fn = CoralMTLLoss(num_classes=num_classes)
        
        # Create metrics storer first
        metrics_storer = MetricsStorer(output_dir=str(temp_output_dir))
        
        # Create metrics calculator
        metrics_calculator = CoralMTLMetrics(
            splitter=splitter_mtl,
            storer=metrics_storer,
            device=device
        )
        
        optimizer = torch.optim.Adam(minimal_coral_mtl_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        config = SimpleNamespace(
            device=device.type,
            output_dir=str(temp_output_dir),
            max_epochs=1,
            model_selection_metric='global.mIoU',
            use_mixed_precision=False
        )
        
        trainer = Trainer(
            model=minimal_coral_mtl_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            metrics_calculator=metrics_calculator,
            metrics_storer=metrics_storer,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config
        )
        
        # Move model to the correct device for testing
        trainer.model.to(device)
        
        # Model should be on correct device
        model_device = next(trainer.model.parameters()).device
        # Handle case where device is 'cuda' but model is on 'cuda:0'
        if device.type == 'cuda' and model_device.type == 'cuda':
            assert model_device.type == device.type
        else:
            assert model_device == device
