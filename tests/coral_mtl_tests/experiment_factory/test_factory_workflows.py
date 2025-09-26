"""High-level workflow tests for `ExperimentFactory` integration."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest
from unittest.mock import patch

from coral_mtl.ExperimentFactory import ExperimentFactory
from coral_mtl.engine.gradient_strategies import NashMTLStrategy
from coral_mtl.engine.pcgrad import PCGrad
from coral_mtl.engine.losses import CoralLoss


@dataclass
class _DummyLoader:
    length: int = 2

    def __len__(self):  # pragma: no cover - trivial
        return self.length

    def __iter__(self):  # pragma: no cover - defensive
        for _ in range(self.length):
            yield None


@pytest.mark.parametrize(
    "strategy_cfg,expected",
    [
        ({'type': 'Uncertainty'}, 'loss'),
        ({'type': 'NashMTL', 'params': {'solver': 'iterative', 'update_frequency': 3}}, 'gradient'),
    ],
)
def test_run_training_configures_trainer_correctly(strategy_cfg, expected, factory_config_dict_mtl):
    cfg = copy.deepcopy(factory_config_dict_mtl)
    cfg['loss']['weighting_strategy'] = strategy_cfg
    cfg.setdefault('metrics_processor', {})['enabled'] = False
    cfg['trainer']['device'] = 'cpu'

    factory = ExperimentFactory(config_dict=cfg)
    loaders = {'train': _DummyLoader(), 'validation': _DummyLoader(), 'test': _DummyLoader()}

    with patch.object(factory, 'get_dataloaders', return_value=loaders), \
            patch('coral_mtl.ExperimentFactory.Trainer') as mock_trainer:
        factory.run_training()

    mock_trainer.assert_called_once()
    trainer_kwargs = mock_trainer.call_args.kwargs
    assert trainer_kwargs['config'].strategy_type == expected


def test_run_training_with_pcgrad_integration(factory_config_dict_mtl):
    cfg = copy.deepcopy(factory_config_dict_mtl)
    cfg['loss']['weighting_strategy'] = {'type': 'NashMTL', 'params': {'solver': 'iterative', 'update_frequency': 2}}
    cfg['optimizer']['use_pcgrad_wrapper'] = True
    cfg.setdefault('metrics_processor', {})['enabled'] = False
    cfg['trainer']['device'] = 'cpu'

    factory = ExperimentFactory(config_dict=cfg)
    loaders = {'train': _DummyLoader(), 'validation': _DummyLoader(), 'test': _DummyLoader()}

    with patch.object(factory, 'get_dataloaders', return_value=loaders), \
            patch('coral_mtl.ExperimentFactory.Trainer') as mock_trainer:
        factory.run_training()

    trainer_kwargs = mock_trainer.call_args.kwargs
    assert isinstance(trainer_kwargs['optimizer'], PCGrad)
    assert isinstance(trainer_kwargs['loss_fn'].weighting_strategy, NashMTLStrategy)
    assert trainer_kwargs['config'].strategy_type == 'gradient'


def test_run_evaluation_passes_loss_fn(factory_config_dict_baseline):
    cfg = copy.deepcopy(factory_config_dict_baseline)
    cfg.setdefault('metrics_processor', {})['enabled'] = False
    cfg['trainer']['device'] = 'cpu'

    factory = ExperimentFactory(config_dict=cfg)
    loaders = {'train': _DummyLoader(), 'validation': _DummyLoader(), 'test': _DummyLoader()}

    with patch.object(factory, 'get_dataloaders', return_value=loaders), \
            patch('coral_mtl.ExperimentFactory.Evaluator') as mock_evaluator:
        factory.run_evaluation()

    evaluator_kwargs = mock_evaluator.call_args.kwargs
    assert isinstance(evaluator_kwargs['loss_fn'], CoralLoss)