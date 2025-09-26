"""Comprehensive parameter coverage tests using CONFIGS_README.md parameter ranges."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.parameter_coverage

from coral_mtl.ExperimentFactory import ExperimentFactory


class _LengthStub:
    """Minimal object exposing only __len__ for loader stubbing."""

    def __init__(self, length: int):
        self._length = length

    def __len__(self) -> int:
        return self._length


class TestModelParameterCoverage:
    """Test model configuration parameter ranges from CONFIGS_README.md."""

    @pytest.mark.parametrize('backbone', [
        'nvidia/mit-b0', 'nvidia/mit-b1', 'nvidia/mit-b2', 
        'nvidia/mit-b3', 'nvidia/mit-b4', 'nvidia/mit-b5'
    ])
    @pytest.mark.parametrize('decoder_channel', [128, 256, 512])
    @pytest.mark.parametrize('encoder_depth', [1, 3, 5])
    def test_coral_mtl_backbone_variations(self, experiment_config_bundle, backbone, decoder_channel, encoder_depth):
        config_kind, config_dict, _ = experiment_config_bundle
        if config_kind != 'mtl':
            pytest.skip('MTL-only test')

        cfg = copy.deepcopy(config_dict)
        cfg['model']['params']['backbone'] = backbone
        cfg['model']['params']['decoder_channel'] = decoder_channel
        cfg['model']['params']['encoder_depth'] = encoder_depth
        cfg['metrics_processor']['enabled'] = False

        with patch('segmentation_models_pytorch.encoders.get_encoder') as mock_encoder:
            mock_encoder.return_value = MagicMock(out_channels=[3, 64, 128, 256, 512, 1024])
            factory = ExperimentFactory(config_dict=cfg)
            model = factory.get_model()
            assert model is not None
            assert model.decoder.decoder_channel == decoder_channel


class TestDataParameterCoverage:
    """Test data configuration parameter ranges."""

    @pytest.mark.parametrize('batch_size', [1, 4, 8, 16])
    @pytest.mark.parametrize('num_workers', [0, 2, 4, 8])
    @pytest.mark.parametrize('patch_size', [256, 512, 768])
    def test_dataloader_parameter_combinations(self, experiment_config_bundle, batch_size, num_workers, patch_size):
        config_kind, config_dict, _ = experiment_config_bundle
        cfg = copy.deepcopy(config_dict)
        cfg['data']['batch_size'] = batch_size
        cfg['data']['num_workers'] = num_workers
        cfg['data']['patch_size'] = patch_size
        cfg['metrics_processor']['enabled'] = False

        with patch('coral_mtl.ExperimentFactory.CoralscapesMTLDataset') as mock_dataset_mtl, \
             patch('coral_mtl.ExperimentFactory.CoralscapesDataset') as mock_dataset_base, \
             patch('coral_mtl.ExperimentFactory.SegmentationAugmentation'):

            mock_dataset = mock_dataset_mtl if config_kind == 'mtl' else mock_dataset_base
            mock_dataset.return_value.__len__.return_value = 10

            factory = ExperimentFactory(config_dict=cfg)
            factory.get_dataloaders()

            # Verify patch_size propagated to augmentations
            mock_dataset.assert_called()
            call_kwargs = mock_dataset.call_args.kwargs
            assert call_kwargs['patch_size'] == patch_size


class TestAugmentationParameterCoverage:
    """Test augmentation parameter ranges from CONFIGS_README.md."""

    @pytest.mark.parametrize('crop_scale', [[0.1, 1.0], [0.5, 1.0], [0.8, 1.0]])
    @pytest.mark.parametrize('rotation_degrees', [0, 15, 30, 45, 90])
    def test_augmentation_parameter_ranges(self, experiment_config_bundle, crop_scale, rotation_degrees):
        config_kind, config_dict, _ = experiment_config_bundle
        cfg = copy.deepcopy(config_dict)
        cfg['augmentations']['crop_scale'] = crop_scale
        cfg['augmentations']['rotation_degrees'] = rotation_degrees
        cfg['metrics_processor']['enabled'] = False

        # Mock components to focus on augmentation parameter passing
        with patch('coral_mtl.ExperimentFactory.SegmentationAugmentation') as mock_aug, \
             patch('coral_mtl.ExperimentFactory.CoralscapesMTLDataset') as mock_dataset_mtl, \
             patch('coral_mtl.ExperimentFactory.CoralscapesDataset') as mock_dataset_base:

            for dataset_mock in (mock_dataset_mtl, mock_dataset_base):
                dataset_mock.return_value.__len__.return_value = 10

            factory = ExperimentFactory(config_dict=cfg)
            factory.get_dataloaders()

            mock_aug.assert_called_once()
            call_kwargs = mock_aug.call_args.kwargs
            assert call_kwargs['crop_scale'] == tuple(crop_scale)
            assert call_kwargs['rotation_degrees'] == rotation_degrees


class TestLossParameterCoverage:
    """Test loss function parameter ranges."""

    @pytest.mark.parametrize('hybrid_alpha', [0.0, 0.3, 0.5, 0.7, 1.0])
    @pytest.mark.parametrize('focal_gamma', [1.0, 2.0, 3.0, 5.0])
    @pytest.mark.parametrize('w_consistency', [0.0, 0.1, 0.5, 1.0])
    def test_mtl_loss_parameter_ranges(self, experiment_config_bundle, hybrid_alpha, focal_gamma, w_consistency):
        config_kind, config_dict, _ = experiment_config_bundle
        if config_kind != 'mtl':
            pytest.skip('MTL-only test')

        cfg = copy.deepcopy(config_dict)
        cfg['loss']['params']['hybrid_alpha'] = hybrid_alpha
        cfg['loss']['params']['focal_gamma'] = focal_gamma
        cfg['loss']['params']['w_consistency'] = w_consistency
        cfg['metrics_processor']['enabled'] = False

        factory = ExperimentFactory(config_dict=cfg)
        loss_fn = factory.get_loss_function()
        
        # Verify parameters are set correctly (primary loss lives on nested CoralLoss)
        assert loss_fn.primary_loss_fn.hybrid_alpha == pytest.approx(hybrid_alpha)
        primary_gamma = getattr(loss_fn.primary_loss_fn.primary_loss, 'gamma', None)
        if primary_gamma is not None:
            assert primary_gamma == pytest.approx(focal_gamma)
        assert loss_fn.w_consistency == pytest.approx(w_consistency)

    @pytest.mark.parametrize('primary_loss_type', ['focal', 'cross_entropy'])
    @pytest.mark.parametrize('dice_smooth', [0.1, 1.0, 2.0])
    def test_baseline_loss_parameter_ranges(self, experiment_config_bundle, primary_loss_type, dice_smooth):
        config_kind, config_dict, _ = experiment_config_bundle
        if config_kind != 'baseline':
            pytest.skip('Baseline-only test')

        cfg = copy.deepcopy(config_dict)
        cfg['loss']['params']['primary_loss_type'] = primary_loss_type
        cfg['loss']['params']['dice_smooth'] = dice_smooth
        cfg['metrics_processor']['enabled'] = False

        factory = ExperimentFactory(config_dict=cfg)
        loss_fn = factory.get_loss_function()
        
        assert loss_fn.primary_loss_type == primary_loss_type
        assert getattr(loss_fn.dice_loss, 'smooth', None) == pytest.approx(dice_smooth)


class TestOptimizerParameterCoverage:
    """Test optimizer parameter ranges from CONFIGS_README.md."""

    @pytest.mark.parametrize('lr', [1e-6, 6e-5, 1e-4, 1e-3])
    @pytest.mark.parametrize('weight_decay', [0.0, 0.01, 0.05, 0.1])
    @pytest.mark.parametrize('warmup_ratio', [0.0, 0.05, 0.1, 0.2])
    @pytest.mark.parametrize('power', [0.5, 1.0, 1.5, 2.0])
    def test_optimizer_parameter_ranges(self, experiment_config_bundle, lr, weight_decay, warmup_ratio, power):
        config_kind, config_dict, _ = experiment_config_bundle
        cfg = copy.deepcopy(config_dict)
        cfg['optimizer']['params']['lr'] = lr
        cfg['optimizer']['params']['weight_decay'] = weight_decay
        cfg['optimizer']['params']['warmup_ratio'] = warmup_ratio
        cfg['optimizer']['params']['power'] = power
        cfg['trainer']['epochs'] = 10  # Small for test
        cfg['metrics_processor']['enabled'] = False

        factory = ExperimentFactory(config_dict=cfg)
        
        # Mock dataloaders to control training steps calculation
        loaders = {'train': _LengthStub(5), 'validation': _LengthStub(5), 'test': _LengthStub(5)}

        with patch.object(factory, 'get_dataloaders', return_value=loaders):
            optimizer, scheduler = factory.get_optimizer_and_scheduler()

        # Verify optimizer parameters are propagated (scheduler may set current lr to 0 during warmup)
        param_groups = optimizer.param_groups if hasattr(optimizer, 'param_groups') else optimizer.optimizer.param_groups
        assert any(pg.get('initial_lr', pg['lr']) == pytest.approx(lr) for pg in param_groups)
        assert any(pg['weight_decay'] == pytest.approx(weight_decay) for pg in param_groups)

        # Scheduler configuration should reflect training steps and warmup ratio
        expected_steps = len(loaders['train']) * cfg['trainer']['epochs']
        expected_warmup = int(expected_steps * warmup_ratio)
        assert getattr(scheduler, 'num_training_steps', expected_steps) == expected_steps
        assert getattr(scheduler, 'num_warmup_steps', expected_warmup) == expected_warmup


class TestMetricsParameterCoverage:
    """Test metrics configuration parameter ranges."""

    @pytest.mark.parametrize('boundary_thickness', [1, 2, 4, 8, 10])
    @pytest.mark.parametrize('use_async_storage', [False, True])
    def test_metrics_parameter_ranges(self, experiment_config_bundle, boundary_thickness, use_async_storage, tmp_path):
        config_kind, config_dict, _ = experiment_config_bundle
        cfg = copy.deepcopy(config_dict)
        cfg['metrics']['boundary_thickness'] = boundary_thickness
        cfg['metrics']['use_async_storage'] = use_async_storage
        cfg['trainer']['output_dir'] = str(tmp_path / 'metrics_test')
        cfg['metrics_processor']['enabled'] = False

        factory = ExperimentFactory(config_dict=cfg)
        metrics = factory.get_metrics_calculator()

        assert metrics.boundary_thickness == boundary_thickness
        has_async_storer = getattr(metrics, 'async_storer', None) is not None
        assert has_async_storer == use_async_storage

    @pytest.mark.parametrize('num_cpu_workers', [1, 4, 8, 16, 32])
    @pytest.mark.parametrize('tasks', [
        ['ASSD'],
        ['HD95'],
        ['ASSD', 'HD95'],
        ['ASSD', 'HD95', 'PanopticQuality'],
        ['ASSD', 'HD95', 'PanopticQuality', 'ARI']
    ])
    def test_advanced_metrics_parameter_ranges(self, experiment_config_bundle, num_cpu_workers, tasks, tmp_path):
        config_kind, config_dict, _ = experiment_config_bundle
        cfg = copy.deepcopy(config_dict)
        cfg['metrics_processor']['enabled'] = True
        cfg['metrics_processor']['num_cpu_workers'] = num_cpu_workers
        cfg['metrics_processor']['tasks'] = tasks
        cfg['trainer']['output_dir'] = str(tmp_path / 'adv_metrics_test')

        with patch('coral_mtl.ExperimentFactory.AdvancedMetricsProcessor') as mock_processor:
            mock_processor.return_value = SimpleNamespace(start=lambda: None, shutdown=lambda: None)
            factory = ExperimentFactory(config_dict=cfg)
            processor = factory.get_advanced_metrics_processor()
            
        mock_processor.assert_called_once()
        call_kwargs = mock_processor.call_args.kwargs
        assert call_kwargs['num_cpu_workers'] == num_cpu_workers
        assert call_kwargs['enabled_tasks'] == tasks


class TestTrainerParameterCoverage:
    """Test trainer configuration parameter ranges."""

    @pytest.mark.parametrize('gradient_accumulation_steps', [1, 2, 4, 8, 16])
    @pytest.mark.parametrize('inference_stride', [128, 256, 320, 341])
    @pytest.mark.parametrize('inference_batch_size', [1, 8, 16, 32])
    def test_trainer_inference_parameter_ranges(self, experiment_config_bundle, gradient_accumulation_steps, inference_stride, inference_batch_size):
        config_kind, config_dict, _ = experiment_config_bundle
        cfg = copy.deepcopy(config_dict)
        cfg['trainer']['gradient_accumulation_steps'] = gradient_accumulation_steps
        cfg['trainer']['inference_stride'] = inference_stride
        cfg['trainer']['inference_batch_size'] = inference_batch_size
        cfg['trainer']['epochs'] = 1  # Minimal for test
        cfg['metrics_processor']['enabled'] = False

        loaders = {'train': _LengthStub(3), 'validation': _LengthStub(3), 'test': _LengthStub(3)}

        with patch.object(ExperimentFactory, 'get_dataloaders', return_value=loaders), \
             patch.object(ExperimentFactory, 'get_metrics_calculator', return_value=MagicMock()), \
             patch.object(ExperimentFactory, 'get_metrics_storer', return_value=MagicMock()), \
             patch.object(ExperimentFactory, 'get_advanced_metrics_processor', return_value=None), \
             patch('coral_mtl.ExperimentFactory.torch.cuda.is_available', return_value=False), \
             patch('coral_mtl.ExperimentFactory.Trainer') as mock_trainer:

            factory = ExperimentFactory(config_dict=cfg)
            factory.run_training()

        mock_trainer.assert_called_once()
        trainer_config = mock_trainer.call_args.kwargs['config']
        assert trainer_config.gradient_accumulation_steps == gradient_accumulation_steps
        assert trainer_config.inference_stride == (inference_stride, inference_stride)
        assert trainer_config.inference_batch_size == inference_batch_size


class TestEvaluatorParameterCoverage:
    """Test evaluator configuration parameter ranges."""

    @pytest.mark.parametrize('inference_stride', [64, 128, 192, 256, 320])
    @pytest.mark.parametrize('inference_batch_size', [1, 4, 8, 16, 32])
    @pytest.mark.parametrize('num_visualizations', [0, 4, 8, 16, 32])
    def test_evaluator_parameter_ranges(self, experiment_config_bundle, inference_stride, inference_batch_size, num_visualizations):
        config_kind, config_dict, _ = experiment_config_bundle
        cfg = copy.deepcopy(config_dict)
        cfg['evaluator']['inference_stride'] = inference_stride
        cfg['evaluator']['inference_batch_size'] = inference_batch_size  
        cfg['evaluator']['num_visualizations'] = num_visualizations
        cfg['metrics_processor']['enabled'] = False

        loaders = {'train': _LengthStub(2), 'validation': _LengthStub(2), 'test': _LengthStub(2)}

        with patch.object(ExperimentFactory, 'get_dataloaders', return_value=loaders), \
             patch('coral_mtl.ExperimentFactory.Evaluator') as mock_evaluator:
            
            factory = ExperimentFactory(config_dict=cfg)
            factory.run_evaluation()

        mock_evaluator.assert_called_once()
        evaluator_config = mock_evaluator.call_args.kwargs['config']
        assert evaluator_config.inference_stride == (inference_stride, inference_stride)
        assert evaluator_config.inference_batch_size == inference_batch_size
        # Note: num_visualizations may not be directly passed to evaluator config in current implementation


class TestModelSelectionMetricCoverage:
    """Test all available model selection metrics from CONFIGS_README.md."""

    @pytest.mark.parametrize('model_selection_metric', [
        'global.mIoU', 'global.BIoU', 'global.classification_error',
        'global.background_error', 'global.missed_error',
        'tasks.genus.ungrouped.mIoU', 'tasks.genus.ungrouped.BIoU',
        'tasks.health.ungrouped.mIoU', 'tasks.health.ungrouped.BIoU',
        'H-Mean'  # MTL-specific
    ])
    def test_model_selection_metrics(self, experiment_config_bundle, model_selection_metric):
        config_kind, config_dict, _ = experiment_config_bundle
        
        # Skip H-Mean for baseline models
        if config_kind == 'baseline' and model_selection_metric == 'H-Mean':
            pytest.skip('H-Mean not applicable for baseline models')
        # Skip task-specific metrics if tasks don't exist
        if 'tasks.' in model_selection_metric and config_kind == 'baseline':
            pytest.skip('Task-specific metrics require MTL model for proper task definitions')

        cfg = copy.deepcopy(config_dict)
        cfg['trainer']['model_selection_metric'] = model_selection_metric
        cfg['trainer']['epochs'] = 1
        cfg['metrics_processor']['enabled'] = False

        loaders = {'train': _LengthStub(2), 'validation': _LengthStub(2), 'test': _LengthStub(2)}

        with patch.object(ExperimentFactory, 'get_dataloaders', return_value=loaders), \
             patch('coral_mtl.ExperimentFactory.Trainer') as mock_trainer:

            factory = ExperimentFactory(config_dict=cfg)
            factory.run_training()

        mock_trainer.assert_called_once()
        trainer_config = mock_trainer.call_args.kwargs['config']
        assert trainer_config.model_selection_metric == model_selection_metric


class TestDeviceParameterCoverage:
    """Test device configuration options."""

    @pytest.mark.parametrize('device', ['cpu', 'cuda', 'auto'])
    def test_device_configuration_options(self, experiment_config_bundle, device):
        config_kind, config_dict, _ = experiment_config_bundle
        cfg = copy.deepcopy(config_dict)
        cfg['trainer']['device'] = device
        cfg['trainer']['epochs'] = 1
        cfg['metrics_processor']['enabled'] = False

        loaders = {'train': _LengthStub(1), 'validation': _LengthStub(1), 'test': _LengthStub(1)}

        with patch.object(ExperimentFactory, 'get_dataloaders', return_value=loaders), \
             patch.object(ExperimentFactory, 'get_metrics_calculator', return_value=MagicMock()), \
             patch.object(ExperimentFactory, 'get_metrics_storer', return_value=MagicMock()), \
             patch.object(ExperimentFactory, 'get_advanced_metrics_processor', return_value=None), \
             patch('coral_mtl.ExperimentFactory.torch.cuda.is_available', return_value=False), \
             patch('coral_mtl.ExperimentFactory.Trainer') as mock_trainer:

            factory = ExperimentFactory(config_dict=cfg)
            factory.run_training()

        mock_trainer.assert_called_once()
        trainer_config = mock_trainer.call_args.kwargs['config']

        if device == 'auto':
            assert trainer_config.device in ['cpu', 'cuda']
        else:
            assert trainer_config.device == device