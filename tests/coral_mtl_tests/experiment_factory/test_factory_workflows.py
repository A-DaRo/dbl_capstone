# Create file: tests/coral_mtl/ExperimentFactory/test_factory_workflows.py
import pytest
from pathlib import Path
from unittest.mock import patch

from coral_mtl.ExperimentFactory import ExperimentFactory

class TestFactoryWorkflows:
    """
    Tests for the high-level orchestration methods (`run_training`, `run_evaluation`)
    of the ExperimentFactory. These tests use mocking to verify that the factory
    correctly assembles and delegates to the Trainer and Evaluator.
    """

    @pytest.fixture
    def mtl_factory(self, mtl_config_yaml: Path) -> ExperimentFactory:
        return ExperimentFactory(config_path=str(mtl_config_yaml))

    def test_run_training_orchestration(self, mtl_factory: ExperimentFactory):
        """
        Verify that `run_training` assembles all components from the factory's
        getters and passes them correctly to the Trainer's constructor before
        calling `train()`.
        """
        # Mock the Trainer class itself
        with patch('coral_mtl.ExperimentFactory.Trainer') as mock_trainer_class:
            # Call the workflow method
            mtl_factory.run_training()

            # --- Assertions ---
            # 1. Assert the Trainer was instantiated exactly once.
            mock_trainer_class.assert_called_once()
            
            # 2. Assert the `train` method on the instance was called.
            trainer_instance = mock_trainer_class.return_value
            trainer_instance.train.assert_called_once()
            
            # 3. Verify that the objects passed to the constructor are the same
            #    instances created and cached by the factory.
            init_kwargs = mock_trainer_class.call_args.kwargs
            assert init_kwargs['model'] is mtl_factory.get_model()
            assert init_kwargs['train_loader'] is mtl_factory.get_dataloaders()['train']
            assert init_kwargs['loss_fn'] is mtl_factory.get_loss_function()
            assert init_kwargs['optimizer'] is mtl_factory.get_optimizer_and_scheduler()[0]
            assert init_kwargs['metrics_calculator'] is mtl_factory.get_metrics_calculator()
            assert init_kwargs['metrics_processor'] is mtl_factory.get_advanced_metrics_processor()

    def test_run_evaluation_orchestration(self, mtl_factory: ExperimentFactory):
        """
        Verify that `run_evaluation` assembles components and correctly
        delegates to the Evaluator class.
        """
        # Mock the Evaluator class
        with patch('coral_mtl.ExperimentFactory.Evaluator') as mock_evaluator_class:
            # The default config might not have a checkpoint path, let's add one.
            override_rel_path = Path('dummy/path/best_model.pth')
            mtl_factory.config['evaluator']['checkpoint_path'] = str(override_rel_path)
            
            # Call the workflow method
            mtl_factory.run_evaluation()

            # --- Assertions ---
            mock_evaluator_class.assert_called_once()
            evaluator_instance = mock_evaluator_class.return_value
            evaluator_instance.evaluate.assert_called_once()
            
            init_kwargs = mock_evaluator_class.call_args.kwargs
            assert init_kwargs['model'] is mtl_factory.get_model()
            assert init_kwargs['test_loader'] is mtl_factory.get_dataloaders()['test']
            assert init_kwargs['metrics_calculator'] is mtl_factory.get_metrics_calculator()
            assert init_kwargs['loss_fn'] is not None
            assert init_kwargs['loss_fn'] is mtl_factory.get_loss_function()
            assert 'config' in init_kwargs
            expected_abs_path = str((mtl_factory.root_path / override_rel_path).resolve())
            assert Path(init_kwargs['config'].checkpoint_path) == Path(expected_abs_path)

    def test_run_evaluation_checkpoint_override(self, mtl_factory: ExperimentFactory):
        """
        Verify that the `checkpoint_path` argument to `run_evaluation`
        takes precedence over the path in the config file.
        """
        with patch('coral_mtl.ExperimentFactory.Evaluator') as mock_evaluator_class:
            # Set a path in the config
            mtl_factory.config['evaluator']['checkpoint_path'] = 'path/from/config.pth'
            
            # Call the workflow with a different, overriding path
            override_path = Path('explicit/path/from/argument.pth')
            mtl_factory.run_evaluation(checkpoint_path=str(override_path))

            # Assert that the Evaluator was initialized with the override path
            init_kwargs = mock_evaluator_class.call_args.kwargs
            expected_override = str((mtl_factory.root_path / override_path).resolve())
            assert Path(init_kwargs['config'].checkpoint_path) == Path(expected_override)