"""High-level workflow tests for `ExperimentFactory` run methods with comprehensive coverage."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from unittest.mock import patch

from coral_mtl.ExperimentFactory import ExperimentFactory
from coral_mtl.engine.losses import CoralLoss, CoralMTLLoss
from coral_mtl.engine.gradient_strategies import NashMTLStrategy
from coral_mtl.engine.pcgrad import PCGrad


@dataclass
class _DummyLoader:
    length: int = 2

    def __len__(self):  # pragma: no cover - trivial
        return self.length

    def __iter__(self):  # pragma: no cover - defensive
        for _ in range(self.length):
            yield None


def _prepare_runner_config(config_dict: dict, *, metrics_enabled: bool, use_pcgrad: bool) -> dict:
    cfg = copy.deepcopy(config_dict)
    cfg.setdefault('metrics_processor', {})['enabled'] = metrics_enabled
    cfg['optimizer']['use_pcgrad_wrapper'] = use_pcgrad
    cfg['trainer']['device'] = 'cpu'
    return cfg


@pytest.mark.parametrize('metrics_enabled', [False, True])
@pytest.mark.parametrize('use_pcgrad', [False, True])
def test_run_training_baseline_pipeline(factory_config_kind, experiment_config_bundle, metrics_enabled, use_pcgrad):
    if factory_config_kind != 'baseline':
        pytest.skip('Baseline-only scenario')

    _, config_dict, _ = experiment_config_bundle
    cfg = _prepare_runner_config(config_dict, metrics_enabled=metrics_enabled, use_pcgrad=use_pcgrad)

    factory = ExperimentFactory(config_dict=cfg)
    loaders = {'train': _DummyLoader(), 'validation': _DummyLoader(), 'test': _DummyLoader()}

    with patch.object(factory, 'get_dataloaders', return_value=loaders), \
            patch('coral_mtl.ExperimentFactory.Trainer') as trainer_mock, \
            patch('coral_mtl.ExperimentFactory.AdvancedMetricsProcessor', return_value=SimpleNamespace(start=lambda: None, shutdown=lambda: None)) as processor_mock:
        factory.run_training()

    trainer_mock.assert_called_once()
    trainer_kwargs = trainer_mock.call_args.kwargs
    assert isinstance(trainer_kwargs['loss_fn'], CoralLoss)
    if use_pcgrad:
        assert isinstance(trainer_kwargs['optimizer'], PCGrad)
    else:
        assert isinstance(trainer_kwargs['optimizer'], torch.optim.AdamW)
    assert trainer_kwargs['config'].strategy_type == 'loss'

    if metrics_enabled:
        processor_mock.assert_called_once()
        assert trainer_kwargs['metrics_processor'] is processor_mock.return_value
    else:
        processor_mock.assert_not_called()
        assert trainer_kwargs['metrics_processor'] is None


@pytest.mark.parametrize('metrics_enabled', [False, True])
@pytest.mark.parametrize('use_pcgrad', [False, True])
@pytest.mark.parametrize(
    'strategy_cfg,expected_strategy_cls,expected_type',
    [
        pytest.param({'type': 'Uncertainty'}, None, 'loss', id='uncertainty'),
        pytest.param({'type': 'NashMTL', 'params': {'solver': 'iterative', 'update_frequency': 2}}, NashMTLStrategy, 'gradient', id='nash'),
    ],
)
def test_run_training_mtl_pipeline(factory_config_kind, experiment_config_bundle, metrics_enabled, use_pcgrad, strategy_cfg, expected_strategy_cls, expected_type):
    if factory_config_kind != 'mtl':
        pytest.skip('MTL-only scenario')

    _, config_dict, _ = experiment_config_bundle
    cfg = _prepare_runner_config(config_dict, metrics_enabled=metrics_enabled, use_pcgrad=use_pcgrad)
    cfg['loss']['weighting_strategy'] = strategy_cfg

    factory = ExperimentFactory(config_dict=cfg)
    loaders = {'train': _DummyLoader(), 'validation': _DummyLoader(), 'test': _DummyLoader()}

    with patch.object(factory, 'get_dataloaders', return_value=loaders), \
            patch('coral_mtl.ExperimentFactory.Trainer') as trainer_mock, \
            patch('coral_mtl.ExperimentFactory.AdvancedMetricsProcessor', return_value=SimpleNamespace(start=lambda: None, shutdown=lambda: None)) as processor_mock:
        factory.run_training()

    trainer_mock.assert_called_once()
    trainer_kwargs = trainer_mock.call_args.kwargs
    assert isinstance(trainer_kwargs['loss_fn'], CoralMTLLoss)
    if expected_strategy_cls:
        assert isinstance(trainer_kwargs['loss_fn'].weighting_strategy, expected_strategy_cls)
    if use_pcgrad:
        assert isinstance(trainer_kwargs['optimizer'], PCGrad)
    else:
        assert isinstance(trainer_kwargs['optimizer'], torch.optim.AdamW)
    assert trainer_kwargs['config'].strategy_type == expected_type

    if metrics_enabled:
        processor_mock.assert_called_once()
        assert trainer_kwargs['metrics_processor'] is processor_mock.return_value
    else:
        processor_mock.assert_not_called()
        assert trainer_kwargs['metrics_processor'] is None


@pytest.mark.parametrize('metrics_enabled', [False, True])
def test_run_evaluation_uses_loss_and_metrics(factory_config_kind, experiment_config_bundle, metrics_enabled):
    _, config_dict, _ = experiment_config_bundle
    cfg = copy.deepcopy(config_dict)
    cfg.setdefault('metrics_processor', {})['enabled'] = metrics_enabled
    cfg['trainer']['device'] = 'cpu'

    factory = ExperimentFactory(config_dict=cfg)
    loaders = {'train': _DummyLoader(), 'validation': _DummyLoader(), 'test': _DummyLoader()}

    with patch.object(factory, 'get_dataloaders', return_value=loaders), \
            patch('coral_mtl.ExperimentFactory.Evaluator') as evaluator_mock, \
            patch('coral_mtl.ExperimentFactory.AdvancedMetricsProcessor', return_value=SimpleNamespace(start=lambda: None, shutdown=lambda: None)) as processor_mock:
        factory.run_evaluation()

    evaluator_mock.assert_called_once()
    evaluator_kwargs = evaluator_mock.call_args.kwargs
    if factory_config_kind == 'mtl':
        assert isinstance(evaluator_kwargs['loss_fn'], CoralMTLLoss)
    else:
        assert isinstance(evaluator_kwargs['loss_fn'], CoralLoss)

    if metrics_enabled:
        processor_mock.assert_called_once()
        assert evaluator_kwargs['metrics_processor'] is processor_mock.return_value
    else:
        processor_mock.assert_not_called()
        assert evaluator_kwargs['metrics_processor'] is None