"""Component instantiation tests for `ExperimentFactory` getters."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest
import torch
from unittest.mock import patch

from coral_mtl.ExperimentFactory import ExperimentFactory
from coral_mtl.model.core import BaselineSegformer, CoralMTLModel
from coral_mtl.metrics.metrics import CoralMetrics, CoralMTLMetrics
from coral_mtl.engine.losses import CoralLoss, CoralMTLLoss
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


@pytest.fixture
def dummy_loaders():
    loader = _DummyLoader()
    return {'train': loader, 'validation': loader, 'test': loader}


@pytest.fixture
def mtl_uncertainty_config(factory_config_dict_mtl: dict) -> dict:
    cfg = copy.deepcopy(factory_config_dict_mtl)
    cfg.setdefault('loss', {})['weighting_strategy'] = {'type': 'Uncertainty'}
    cfg.setdefault('optimizer', {})['use_pcgrad_wrapper'] = False
    cfg.setdefault('metrics_processor', {})['enabled'] = False
    return cfg


def test_factory_builds_baseline_components(baseline_config_yaml, dummy_loaders):
    factory = ExperimentFactory(config_path=str(baseline_config_yaml))

    model = factory.get_model()
    loss_fn = factory.get_loss_function()
    with patch.object(factory, 'get_dataloaders', return_value=dummy_loaders):
        optimizer, _ = factory.get_optimizer_and_scheduler()
    metrics = factory.get_metrics_calculator()

    assert isinstance(model, BaselineSegformer)
    assert isinstance(loss_fn, CoralLoss)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(metrics, CoralMetrics)


def test_factory_builds_mtl_components(mtl_uncertainty_config, dummy_loaders):
    factory = ExperimentFactory(config_dict=mtl_uncertainty_config)

    model = factory.get_model()
    loss_fn = factory.get_loss_function()
    metrics = factory.get_metrics_calculator()

    assert isinstance(model, CoralMTLModel)
    assert isinstance(loss_fn, CoralMTLLoss)
    assert isinstance(loss_fn.weighting_strategy, UncertaintyWeightingStrategy)
    assert isinstance(metrics, CoralMTLMetrics)


@pytest.mark.parametrize(
    "strategy_type,expected_cls,params",
    [
        ("Uncertainty", UncertaintyWeightingStrategy, {}),
        ("IMGrad", IMGradStrategy, {'params': {'solver': 'pgd'}}),
        ("NashMTL", NashMTLStrategy, {'params': {'solver': 'iterative', 'update_frequency': 3}}),
    ],
)
def test_factory_builds_correct_strategy(strategy_type, expected_cls, params, factory_config_dict_mtl, dummy_loaders):
    cfg = copy.deepcopy(factory_config_dict_mtl)
    cfg.setdefault('loss', {})['weighting_strategy'] = {'type': strategy_type, **params}
    cfg.setdefault('optimizer', {})['use_pcgrad_wrapper'] = False
    cfg.setdefault('metrics_processor', {})['enabled'] = False

    factory = ExperimentFactory(config_dict=cfg)

    loss_fn = factory.get_loss_function()
    assert isinstance(loss_fn.weighting_strategy, expected_cls)


def test_factory_configures_pcgrad_wrapper(factory_config_dict_mtl, dummy_loaders):
    base_cfg = copy.deepcopy(factory_config_dict_mtl)
    base_cfg.setdefault('metrics_processor', {})['enabled'] = False

    cfg_no_pcgrad = copy.deepcopy(base_cfg)
    cfg_no_pcgrad.setdefault('optimizer', {})['use_pcgrad_wrapper'] = False
    with patch.object(ExperimentFactory, 'get_dataloaders', return_value=dummy_loaders):
        factory_standard = ExperimentFactory(config_dict=cfg_no_pcgrad)
        optimizer_std, _ = factory_standard.get_optimizer_and_scheduler()
    assert isinstance(optimizer_std, torch.optim.AdamW)

    cfg_pcgrad = copy.deepcopy(base_cfg)
    cfg_pcgrad.setdefault('optimizer', {})['use_pcgrad_wrapper'] = True
    with patch.object(ExperimentFactory, 'get_dataloaders', return_value=dummy_loaders):
        factory_pcgrad = ExperimentFactory(config_dict=cfg_pcgrad)
        optimizer_pg, _ = factory_pcgrad.get_optimizer_and_scheduler()
    assert isinstance(optimizer_pg, PCGrad)