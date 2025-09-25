# Create file: tests/coral_mtl/ExperimentFactory/test_factory_getters.py
import pytest
from pathlib import Path
import torch

from coral_mtl.ExperimentFactory import ExperimentFactory
from coral_mtl.model.core import CoralMTLModel, BaselineSegformer
from coral_mtl.metrics.metrics import CoralMTLMetrics, CoralMetrics
from coral_mtl.engine.losses import CoralMTLLoss, CoralLoss
from coral_mtl.metrics.metrics_storer import AdvancedMetricsProcessor

class TestFactoryGetters:
    """Tests for the component-building `get_*` methods of the ExperimentFactory."""

    @pytest.fixture
    def mtl_factory(self, mtl_config_yaml: Path) -> ExperimentFactory:
        return ExperimentFactory(config_path=str(mtl_config_yaml))

    @pytest.fixture
    def baseline_factory(self, baseline_config_yaml: Path) -> ExperimentFactory:
        return ExperimentFactory(config_path=str(baseline_config_yaml))

    @pytest.mark.parametrize("getter_name", [
        "get_model",
        "get_dataloaders",
        "get_loss_function",
        "get_optimizer_and_scheduler",
        "get_metrics_calculator",
        "get_metrics_storer",
        "get_advanced_metrics_processor"
    ])
    def test_caching_behavior(self, mtl_factory: ExperimentFactory, getter_name: str):
        """
        CRITICAL: Verify that all getters cache their results. Calling a getter
        multiple times must return the exact same object instance.
        """
        # For optimizer, which needs the model as an argument
        if getter_name == "get_optimizer_and_scheduler":
            obj1 = getattr(mtl_factory, getter_name)()
            obj2 = getattr(mtl_factory, getter_name)()
        else:
            obj1 = getattr(mtl_factory, getter_name)()
            obj2 = getattr(mtl_factory, getter_name)()
        
        assert obj1 is obj2, f"Getter '{getter_name}' failed to cache its object."

    def test_get_model_types(self, mtl_factory: ExperimentFactory, baseline_factory: ExperimentFactory):
        """Verify the correct model class is instantiated based on the config."""
        assert isinstance(mtl_factory.get_model(), CoralMTLModel)
        assert isinstance(baseline_factory.get_model(), BaselineSegformer)
    
    def test_get_loss_function_types(self, mtl_factory: ExperimentFactory, baseline_factory: ExperimentFactory):
        """Verify the correct loss class is instantiated based on the config."""
        assert isinstance(mtl_factory.get_loss_function(), CoralMTLLoss)
        assert isinstance(baseline_factory.get_loss_function(), CoralLoss)

    def test_get_metrics_calculator_types(self, mtl_factory: ExperimentFactory, baseline_factory: ExperimentFactory):
        """Verify the correct metrics calculator class is instantiated."""
        assert isinstance(mtl_factory.get_metrics_calculator(), CoralMTLMetrics)
        assert isinstance(baseline_factory.get_metrics_calculator(), CoralMetrics)

    def test_get_advanced_metrics_processor_conditional(self, factory_config_dict_mtl: dict):
        """
        Verify the AdvancedMetricsProcessor is only created when `enabled: true`
        in the configuration.
        """
        # Case 1: Enabled is true
        config_enabled = factory_config_dict_mtl.copy()
        config_enabled['metrics_processor'] = {'enabled': True, 'num_cpu_workers': 2}
        factory_enabled = ExperimentFactory(config_dict=config_enabled)
        processor = factory_enabled.get_advanced_metrics_processor()
        assert isinstance(processor, AdvancedMetricsProcessor)

        # Case 2: Enabled is false
        config_disabled = factory_config_dict_mtl.copy()
        config_disabled['metrics_processor'] = {'enabled': False}
        factory_disabled = ExperimentFactory(config_dict=config_disabled)
        assert factory_disabled.get_advanced_metrics_processor() is None

        # Case 3: Key is missing entirely
        config_missing = factory_config_dict_mtl.copy()
        if 'metrics_processor' in config_missing:
            del config_missing['metrics_processor']
        factory_missing = ExperimentFactory(config_dict=config_missing)
        assert factory_missing.get_advanced_metrics_processor() is None