"""Component instantiation tests for `ExperimentFactory` getters with parameter coverage."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from unittest.mock import patch

from coral_mtl.ExperimentFactory import ExperimentFactory
from coral_mtl.model.core import BaselineSegformer, CoralMTLModel
from coral_mtl.metrics.metrics import CoralMetrics, CoralMTLMetrics
from coral_mtl.metrics.metrics_storer import MetricsStorer
from coral_mtl.engine.losses import CoralLoss
from coral_mtl.engine.loss_weighting import (
    UncertaintyWeightingStrategy,
)
from coral_mtl.engine.gradient_strategies import IMGradStrategy, NashMTLStrategy
from coral_mtl.engine.pcgrad import PCGrad


@dataclass
class _DummyLoader:
    length: int = 2

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.length

    def __iter__(self):  # pragma: no cover - defensive
        for _ in range(self.length):
            yield None


def _prepare_config(config_dict: dict) -> dict:
    """Return a deep copy of the config with safe defaults for testing."""
    cfg = copy.deepcopy(config_dict)
    cfg.setdefault('metrics_processor', {})['enabled'] = False
    cfg.setdefault('data', {}).setdefault('num_workers', 0)
    return cfg


@pytest.mark.parametrize('factory_section_name', ['model'], indirect=True)
def test_get_model_aligns_with_task_definitions(factory_section_config, experiment_config_bundle):
    config_kind, section_name, section = factory_section_config
    bundle_kind, config_dict, _ = experiment_config_bundle
    assert config_kind == bundle_kind

    cfg = _prepare_config(config_dict)
    factory = ExperimentFactory(config_dict=cfg)
    model = factory.get_model()

    model.eval()  # Set to eval mode to avoid BatchNorm issues with small tensors
    
    if config_kind == 'mtl':
        assert isinstance(model, CoralMTLModel)
        expected_tasks = cfg['model']['tasks']['primary'] + cfg['model']['tasks']['auxiliary']
        dummy_images = torch.rand(1, 3, 128, 128)  # Larger tensor to avoid BatchNorm issues
        outputs = model(dummy_images)
        assert set(outputs.keys()) == set(expected_tasks)
        for task, logits in outputs.items():
            task_info = factory.task_splitter.hierarchical_definitions[task]
            expected_classes = len(task_info['grouped']['id2label']) if task_info['is_grouped'] else len(task_info['ungrouped']['id2label'])
            assert logits.shape[1] == expected_classes
    else:
        assert isinstance(model, BaselineSegformer)
        dummy_images = torch.rand(1, 3, 128, 128)  # Larger tensor to avoid BatchNorm issues
        outputs = model(dummy_images)
        expected_classes = len(factory.task_splitter.global_id2label)
        assert outputs.shape[1] == expected_classes


@pytest.mark.parametrize('factory_section_name', ['data'], indirect=True)
def test_get_dataloaders_selects_correct_dataset(factory_section_config, experiment_config_bundle):
    config_kind, _, _ = factory_section_config
    bundle_kind, config_dict, _ = experiment_config_bundle
    assert config_kind == bundle_kind

    cfg = _prepare_config(config_dict)
    cfg['data'].setdefault('batch_size_per_gpu', cfg['data'].get('batch_size', 1))

    dataset_module_path = (
        'coral_mtl.ExperimentFactory.CoralscapesMTLDataset'
        if config_kind == 'mtl'
        else 'coral_mtl.ExperimentFactory.CoralscapesDataset'
    )

    created_splits = []

    class DummyDataset:
        def __init__(self, split, **_):
            self.split = split

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {'image': torch.zeros(3, 32, 32), 'mask': torch.zeros(32, 32)}

    def dataset_constructor(*args, **kwargs):
        created_splits.append(kwargs['split'])
        return DummyDataset(split=kwargs['split'])

    def dataloader_constructor(dataset, **kwargs):
        return SimpleNamespace(dataset=dataset, kwargs=kwargs)

    with patch(dataset_module_path, side_effect=dataset_constructor) as dataset_mock, \
            patch('coral_mtl.ExperimentFactory.DataLoader', side_effect=dataloader_constructor), \
            patch('coral_mtl.ExperimentFactory.SegmentationAugmentation', autospec=True):
        factory = ExperimentFactory(config_dict=cfg)
        loaders = factory.get_dataloaders()

    assert dataset_mock.call_count == 3
    assert created_splits == ['train', 'validation', 'test']
    assert set(loaders.keys()) == {'train', 'validation', 'test'}
    for split, loader in loaders.items():
        assert isinstance(loader.dataset, DummyDataset)
        assert loader.dataset.split == split


@pytest.mark.parametrize('factory_section_name', ['optimizer'], indirect=True)
@pytest.mark.parametrize('use_pcgrad', [False, True])
def test_get_optimizer_and_scheduler_toggle_pcgrad(factory_section_config, experiment_config_bundle, use_pcgrad):
    config_kind, _, _ = factory_section_config
    bundle_kind, config_dict, _ = experiment_config_bundle
    assert config_kind == bundle_kind

    cfg = _prepare_config(config_dict)
    cfg['optimizer']['use_pcgrad_wrapper'] = use_pcgrad
    cfg['trainer']['epochs'] = 2

    factory = ExperimentFactory(config_dict=cfg)
    dummy_loader = _DummyLoader(length=3)
    loaders = {'train': dummy_loader, 'validation': dummy_loader, 'test': dummy_loader}

    with patch.object(factory, 'get_dataloaders', return_value=loaders):
        optimizer, scheduler = factory.get_optimizer_and_scheduler()

    if use_pcgrad:
        assert isinstance(optimizer, PCGrad)
    else:
        assert isinstance(optimizer, torch.optim.AdamW)
    assert scheduler is not None
    # Cached tuple should be reused
    with patch.object(factory, 'get_dataloaders', side_effect=AssertionError("should use cache")):
        cached_optimizer, cached_scheduler = factory.get_optimizer_and_scheduler()
    assert optimizer is cached_optimizer
    assert scheduler is cached_scheduler


@pytest.mark.parametrize('factory_section_name', ['loss'], indirect=True)
def test_get_loss_function_baseline_returns_hybrid(factory_section_config, experiment_config_bundle):
    config_kind, _, _ = factory_section_config
    if config_kind != 'baseline':
        pytest.skip('Baseline-only assertion')

    _, config_dict, _ = experiment_config_bundle
    cfg = _prepare_config(config_dict)
    factory = ExperimentFactory(config_dict=cfg)
    loss_fn = factory.get_loss_function()
    assert isinstance(loss_fn, CoralLoss)
    assert factory.get_loss_function() is loss_fn


@pytest.mark.parametrize('factory_section_name', ['loss'], indirect=True)
@pytest.mark.parametrize(
    'strategy_cfg,expected_cls',
    [
        ({'type': 'Uncertainty'}, UncertaintyWeightingStrategy),
        ({'type': 'IMGrad', 'params': {'solver': 'pgd'}}, IMGradStrategy),
        ({'type': 'NashMTL', 'params': {'solver': 'iterative', 'update_frequency': 3}}, NashMTLStrategy),
    ],
)
def test_get_loss_function_mtl_weighting(factory_section_config, experiment_config_bundle, strategy_cfg, expected_cls):
    config_kind, _, _ = factory_section_config
    if config_kind != 'mtl':
        pytest.skip('MTL-only assertion')

    _, config_dict, _ = experiment_config_bundle
    cfg = _prepare_config(config_dict)
    cfg['loss']['weighting_strategy'] = strategy_cfg
    cfg['optimizer']['use_pcgrad_wrapper'] = False

    factory = ExperimentFactory(config_dict=cfg)
    loss_fn = factory.get_loss_function()
    assert isinstance(loss_fn, CoralLoss)
    assert isinstance(loss_fn.weighting_strategy, expected_cls)


@pytest.mark.parametrize('factory_section_name', ['metrics'], indirect=True)
def test_get_metrics_calculator_matches_model_type(factory_section_config, experiment_config_bundle, tmp_path):
    config_kind, _, _ = factory_section_config
    bundle_kind, config_dict, _ = experiment_config_bundle
    assert config_kind == bundle_kind

    cfg = _prepare_config(config_dict)
    cfg['trainer']['output_dir'] = str(tmp_path / f"{config_kind}_metrics")
    factory = ExperimentFactory(config_dict=cfg)
    metrics = factory.get_metrics_calculator()

    if config_kind == 'mtl':
        assert isinstance(metrics, CoralMTLMetrics)
    else:
        assert isinstance(metrics, CoralMetrics)
    # Cached object reused
    assert factory.get_metrics_calculator() is metrics


@pytest.mark.parametrize('factory_section_name', ['trainer'], indirect=True)
def test_get_metrics_storer_resolves_output_dir(factory_section_config, experiment_config_bundle, tmp_path):
    config_kind, _, _ = factory_section_config
    bundle_kind, config_dict, _ = experiment_config_bundle
    assert config_kind == bundle_kind

    cfg = _prepare_config(config_dict)
    relative_dir = tmp_path / 'relative_run'
    cfg['trainer']['output_dir'] = relative_dir.as_posix()

    factory = ExperimentFactory(config_dict=cfg)
    storer = factory.get_metrics_storer()
    assert isinstance(storer, MetricsStorer)
    assert Path(storer.output_dir).is_absolute()
    assert factory.get_metrics_storer() is storer


@pytest.mark.parametrize('factory_section_name', ['metrics_processor'], indirect=True)
@pytest.mark.parametrize('enabled', [False, True])
def test_get_advanced_metrics_processor_toggle(factory_section_config, experiment_config_bundle, tmp_path, enabled):
    config_kind, _, _ = factory_section_config
    bundle_kind, config_dict, _ = experiment_config_bundle
    assert config_kind == bundle_kind

    cfg = copy.deepcopy(config_dict)
    cfg.setdefault('metrics_processor', {})['enabled'] = enabled
    cfg['trainer']['output_dir'] = str(tmp_path / f"{config_kind}_adv_metrics")

    with patch('coral_mtl.ExperimentFactory.AdvancedMetricsProcessor') as processor_mock:
        processor_mock.return_value = SimpleNamespace(start=lambda: None, shutdown=lambda: None)
        factory = ExperimentFactory(config_dict=cfg)
        processor = factory.get_advanced_metrics_processor()

    if enabled:
        processor_mock.assert_called_once()
        assert processor is processor_mock.return_value
        # Cached instance reused
        assert factory.get_advanced_metrics_processor() is processor
    else:
        processor_mock.assert_not_called()
        assert processor is None