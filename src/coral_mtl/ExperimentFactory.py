import os
import yaml
import math
import optuna
import copy
import torch
import torch.nn as nn
from torch.optim import Optimizer
from types import SimpleNamespace
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the model classes defined in the project structure
from .model.core import CoralMTLModel, BaselineSegformer
from .data.dataset import CoralscapesMTLDataset, CoralscapesDataset
from .data.augmentations import SegmentationAugmentation
from .engine.optimizer import create_optimizer_and_scheduler
from .engine.losses import CoralMTLLoss, CoralLoss
from .engine.metrics import AbstractCoralMetrics, CoralMTLMetrics, CoralMetrics
from .engine.trainer import Trainer
from .engine.evaluator import Evaluator
from .utils.visualization import Visualizer


class ExperimentFactory:
    """
    The master factory and orchestrator for all experimental workflows.
    """

    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """
        The constructor for the ExperimentFactory.

        It reads the main configuration file and immediately pre-processes any linked
        configuration files (like task definitions) to prepare for component building.

        Args:
            config_path (str, optional): The path to the main YAML configuration file.
            config_dict (Dict, optional): A dictionary containing the configuration.
        """
        if config_dict:
            self.config = config_dict
        elif config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError("Either 'config_path' or 'config_dict' must be provided.")

        # --- Initialize internal caches ---
        self.model = None
        self.task_info = {} # Will store pre-parsed info from task_definitions.yaml

        # --- Pre-process linked configurations ---
        self._parse_task_definitions()

    def _parse_task_definitions(self):
        """
        A private helper to load and parse the task_definitions.yaml file.
        This centralizes the logic for understanding the hierarchical task structure
        and makes the class counts and label mappings available to all other factory methods.
        """
        task_def_path = self.config.get('data', {}).get('task_definitions_path')
        if not task_def_path:
            return  # No task file specified, nothing to do.

        with open(task_def_path, 'r') as f:
            task_definitions = yaml.safe_load(f)

        # Store the full task definitions, number of classes, and id2label mappings.
        self.task_info['definitions'] = task_definitions

        num_classes_per_task = {}
        id2label_per_task = {}
        for task_name, details in task_definitions.items():
            # Ensure keys are integers for proper indexing
            id2label = {int(k): v for k, v in details['id2label'].items()}
            num_classes_per_task[task_name] = len(id2label)
            id2label_per_task[task_name] = id2label

        self.task_info['num_classes'] = num_classes_per_task
        self.task_info['id2label'] = id2label_per_task

    def get_model(self) -> torch.nn.Module:
        """
        Builds or retrieves the cached model instance.

        This method acts as a factory, reading the 'model' section of the config
        and instantiating the correct model class with the specified parameters.
        For the CoralMTLModel, it dynamically determines the number of classes for
        each decoder head from the pre-parsed task definitions.

        Returns:
            torch.nn.Module: The instantiated PyTorch model.
        """
        # 1. Check cache first to avoid re-building
        if self.model is not None:
            return self.model

        # 2. Read model configuration
        model_config = self.config.get('model', {})
        model_type = model_config.get('type')
        if not model_type:
            raise ValueError("Model 'type' not specified in the configuration file.")

        params = model_config.get('params', {})
        print(f"--- Building model of type: {model_type} ---")

        # 3. Instantiate the correct model based on its type
        if model_type == "CoralMTL":
            # For the MTL model, we dynamically assemble its parameters
            tasks_config = model_config.get('tasks', {})
            primary_tasks = tasks_config.get('primary', [])
            aux_tasks = tasks_config.get('auxiliary', [])
            
            # Get the number of classes for each task from the pre-parsed info
            num_classes_dict = self.task_info.get('num_classes')
            if not num_classes_dict:
                raise ValueError(
                    "CoralMTL model requires 'task_definitions_path' to be set in "
                    "the data config to determine the number of classes for each head."
                )

            model = CoralMTLModel(
                encoder_name=params['backbone'],
                decoder_channel=params['decoder_channel'],
                num_classes=num_classes_dict,
                attention_dim=params['attention_dim'],
                primary_tasks=primary_tasks,
                aux_tasks=aux_tasks
            )

        elif model_type == "SegFormerBaseline":
            # The baseline model has a simpler, static configuration
            model = BaselineSegformer(
                encoder_name=params['backbone'],
                decoder_channel=params['decoder_channel'],
                num_classes=params['num_classes']
            )
            
        else:
            raise ValueError(f"Unknown model type '{model_type}' specified in config.")

        # 4. Cache and return the model
        self.model = model
        return self.model
    
    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Builds or retrieves cached DataLoaders for train, validation, and test splits.

        This method intelligently handles data loading from either local paths or the
        Hugging Face Hub. It instantiates the correct Dataset class (MTL vs. non-MTL)
        based on the model type and applies augmentations only to the training set.

        Returns:
            Dict[str, DataLoader]: A dictionary containing 'train', 'val', and 'test' loaders.
        """
        # 1. Check cache first
        if hasattr(self, 'dataloaders') and self.dataloaders:
            return self.dataloaders

        print("--- Building dataloaders ---")
        data_config = self.config.get('data', {})
        aug_config = self.config.get('augmentations', {})
        model_type = self.config.get('model', {}).get('type')

        # 2. Instantiate augmentations for the training set
        train_augmentations = SegmentationAugmentation(
            patch_size=data_config.get('patch_size', 512),
            crop_scale=tuple(aug_config.get('crop_scale', [0.5, 1.0])),
            rotation_degrees=aug_config.get('rotation_degrees', 15),
            jitter_params=aug_config.get('jitter_params')
        )

        # 3. Determine data source (HF Hub vs. Local)
        # The AbstractCoralscapesDataset will handle the logic internally.
        dataset_name = data_config.get('dataset_name')
        if os.path.isdir(dataset_name):
            # It's a local path, so we don't pass an hf_dataset_name
            hf_dataset_name = None
            # The local path could be either the raw data root or the PDS root
            # but dataset.py expects specific path variables. We provide both.
            data_root_path = data_config.get('data_root_path')
            pds_train_path = data_config.get('pds_train_path')
        else:
            # It's not a valid directory, so assume it's an HF Hub ID.
            hf_dataset_name = dataset_name
            data_root_path = None
            pds_train_path = None
            
        # 4. Prepare shared arguments for all dataset splits
        shared_dataset_args = {
            'patch_size': data_config.get('patch_size', 512),
            'hf_dataset_name': hf_dataset_name,
            'data_root_path': data_root_path,
            'pds_train_path': pds_train_path,
        }

        # 5. Determine which Dataset class to use
        if model_type == "CoralMTL":
            DatasetClass = CoralscapesMTLDataset
            # The MTL dataset requires the parsed task definitions
            specific_args = {'task_definitions': self.task_info.get('definitions')}
        elif model_type == "SegFormerBaseline":
            DatasetClass = CoralscapesDataset
            # The baseline dataset can optionally take the definitions for flattening
            specific_args = {'task_definitions': self.task_info.get('definitions')}
        else:
            raise ValueError(f"Dataloader factory cannot handle unknown model type '{model_type}'.")

        # 6. Create datasets and dataloaders for each split
        self.dataloaders = {}
        for split in ['train', 'val', 'test']:
            # Use augmentations only for the training split
            augs = train_augmentations if split == 'train' else None
            
            # Map 'val' and 'test' splits to Hugging Face's 'validation' and 'test'
            hf_split_name = 'validation' if split == 'val' else split

            dataset = DatasetClass(
                split=hf_split_name,
                augmentations=augs,
                **shared_dataset_args,
                **specific_args
            )
            
            # Use shuffle only for the training dataloader
            shuffle = (split == 'train')
            
            self.dataloaders[split] = DataLoader(
                dataset,
                batch_size=data_config.get('batch_size', 4),
                shuffle=shuffle,
                num_workers=data_config.get('num_workers', 4),
                pin_memory=True
            )
            
        return self.dataloaders
    
    def get_optimizer_and_scheduler(self) -> Tuple[Optimizer, Any]:
        """
        Builds the optimizer and learning rate scheduler.

        This method has a crucial dependency: it must be called *after* the
        dataloaders have been created, as it needs to know the number of
        training steps per epoch to correctly configure the scheduler.

        Returns:
            Tuple[Optimizer, Any]: A tuple containing the instantiated optimizer
                                   and learning rate scheduler.
        """
        # 1. Check cache first
        if hasattr(self, 'optimizer') and self.optimizer:
            return self.optimizer, self.scheduler

        print("--- Building optimizer and scheduler ---")
        optimizer_config = self.config.get('optimizer', {})
        trainer_config = self.config.get('trainer', {})
        optimizer_type = optimizer_config.get('type')

        if not optimizer_type:
            raise ValueError("Optimizer 'type' not specified in the configuration file.")
        
        # 2. Ensure dependencies are met
        model = self.get_model()
        dataloaders = self.get_dataloaders()
        train_loader = dataloaders.get('train')
        if not train_loader:
            raise ValueError("Training dataloader is required to configure the scheduler.")

        # 3. Calculate dynamic scheduler parameters
        num_epochs = trainer_config.get('epochs', 100)
        num_training_steps = len(train_loader) * num_epochs
        
        params = optimizer_config.get('params', {})
        warmup_ratio = params.get('warmup_ratio', 0.1)
        num_warmup_steps = int(num_training_steps * warmup_ratio)

        # 4. Instantiate based on type
        if optimizer_type == "AdamWPolyDecay":
            optimizer, scheduler = create_optimizer_and_scheduler(
                model=model,
                learning_rate=params.get('lr', 6e-5),
                weight_decay=params.get('weight_decay', 0.01),
                adam_betas=tuple(params.get('adam_betas', [0.9, 0.999])),
                num_training_steps=num_training_steps,
                num_warmup_steps=num_warmup_steps,
                power=params.get('power', 1.0)
            )
        else:
            raise ValueError(f"Unknown optimizer type '{optimizer_type}' specified in config.")

        # 5. Cache and return
        self.optimizer = optimizer
        self.scheduler = scheduler
        return self.optimizer, self.scheduler


    def get_loss_function(self) -> nn.Module:
        """
        Builds the loss function.

        Selects the appropriate loss class (MTL vs. non-MTL) based on the
        configuration and injects the necessary parameters.

        Returns:
            nn.Module: The instantiated loss function module.
        """
        # 1. Check cache first
        if hasattr(self, 'loss_fn') and self.loss_fn:
            return self.loss_fn

        print("--- Building loss function ---")
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type')

        if not loss_type:
            # Provide a sensible default for baseline models if no loss is specified
            print("Warning: No loss 'type' specified. Defaulting to CrossEntropyLoss.")
            self.loss_fn = nn.CrossEntropyLoss()
            return self.loss_fn

        params = loss_config.get('params', {})
        
        # 2. Instantiate based on type
        if loss_type == "CompositeHierarchical":
            num_classes_dict = self.task_info.get('num_classes')
            if not num_classes_dict:
                 raise ValueError("CoralMTLLoss requires 'task_definitions_path' to be set.")
            
            loss_fn = CoralMTLLoss(
                num_classes=num_classes_dict,
                ignore_index=params.get('ignore_index', 0),
                w_consistency=params.get('w_consistency', 0.1),
                hybrid_alpha=params.get('hybrid_alpha', 0.5),
                focal_gamma=params.get('focal_gamma', 2.0)
            )

        elif loss_type == "HybridLoss":
            loss_fn = CoralLoss(
                primary_loss_type=params.get('primary_loss_type', 'focal'),
                hybrid_alpha=params.get('hybrid_alpha', 0.5),
                focal_gamma=params.get('focal_gamma', 2.0),
                dice_smooth=params.get('dice_smooth', 1.0),
                ignore_index=params.get('ignore_index', 0)
            )

        else:
            raise ValueError(f"Unknown loss type '{loss_type}' specified in config.")
        
        # 3. Cache and return
        self.loss_fn = loss_fn
        return self.loss_fn
    

    def get_metrics_calculator(self) -> AbstractCoralMetrics:
        """
        Builds the appropriate metrics calculator based on the model type.

        - For MTL models, it uses CoralMTLMetrics, which expects dictionary-based
          predictions and leverages the pre-parsed task information for class counts.
        - For non-MTL (baseline) models, it uses CoralMetrics, which takes the
          task_definitions.yaml file to un-flatten single-mask predictions for a
          fair, hierarchical evaluation.

        Returns:
            AbstractCoralMetrics: The instantiated metrics calculator object.
        """
        # 1. Check cache first
        if hasattr(self, 'metrics_calculator') and self.metrics_calculator:
            return self.metrics_calculator

        print("--- Building metrics calculator ---")
        metrics_config = self.config.get('metrics', {})
        model_type = self.config.get('model', {}).get('type')
        if not model_type:
            raise ValueError("Model 'type' must be specified to build the correct metrics calculator.")

        # 2. Prepare shared parameters common to both calculator classes
        # A real implementation might get the device from a central property.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        shared_params = {
            'device': device,
            'primary_tasks': metrics_config.get('primary_tasks', []),
            'boundary_thickness': metrics_config.get('boundary_thickness', 2),
            'ignore_index': metrics_config.get('ignore_index', 255)
        }

        # 3. Instantiate the correct metrics class based on the model type
        if model_type == "CoralMTL":
            num_classes_dict = self.task_info.get('num_classes')
            if not num_classes_dict:
                 raise ValueError(
                    "CoralMTLMetrics requires 'task_definitions_path' to be set in the "
                    "data config so the factory can determine class counts."
                 )
            
            metrics_calculator = CoralMTLMetrics(
                num_classes=num_classes_dict,
                **shared_params
            )

        elif model_type == "SegFormerBaseline":
            task_definitions = self.task_info.get('definitions')
            if not task_definitions:
                raise ValueError(
                    "CoralMetrics (for baseline models) requires 'task_definitions_path' "
                    "to be set in the data config for hierarchical evaluation."
                )

            metrics_calculator = CoralMetrics(
                task_definitions=task_definitions,
                **shared_params
            )
            
        else:
            raise ValueError(f"Unknown model type '{model_type}' for metrics calculator.")
        
        # 4. Cache and return the instantiated object
        self.metrics_calculator = metrics_calculator
        return self.metrics_calculator

    def run_training(self, trial: Optional[optuna.Trial] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Executes a complete model training and validation workflow.

        This method orchestrates the entire training process. It uses its internal
        getter methods to assemble all necessary components (model, data loaders,
        optimizer, loss function, and metrics calculator) based on the loaded
        configuration. It then instantiates a dedicated `Trainer` object,
        hands off all the components, and initiates the training loop.

        Args:
            trial (Optional[optuna.Trial]): An Optuna trial object, if running as part of a study.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]:
                - A dictionary containing the training history logs.
                - A dictionary containing the validation history logs.
        """
        print("\n" + "="*50)
        print(">>> Starting Training Workflow <<<")
        print("="*50)

        # 1. Assemble all required components using internal getter methods.
        #    This ensures a consistent and config-driven setup.
        print("1/4: Assembling model...")
        model = self.get_model()
        
        print("2/4: Assembling dataloaders...")
        dataloaders = self.get_dataloaders()
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        print("3/4: Assembling optimizer, scheduler, loss, and metrics...")
        optimizer, scheduler = self.get_optimizer_and_scheduler()
        loss_fn = self.get_loss_function()
        metrics_calculator = self.get_metrics_calculator()

        # 2. Prepare the trainer-specific configuration object.
        #    The Trainer class expects an object with attribute access (e.g., config.DEVICE),
        #    so we convert the dictionary from our YAML into a SimpleNamespace.
        trainer_config_dict = self.config.get('trainer', {})
        if 'device' not in trainer_config_dict or trainer_config_dict['device'] == 'auto':
            trainer_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Add patch_size for the inferrer, it's a data-level config but needed by trainer
        trainer_config_dict['PATCH_SIZE'] = self.config.get('data', {}).get('patch_size', 512)
        
        trainer_config = SimpleNamespace(**trainer_config_dict)

        # 3. Instantiate the dedicated Trainer class.
        #    The factory's job is to build and provide all dependencies.
        print("4/4: Initializing the training engine...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            metrics_calculator=metrics_calculator,
            optimizer=optimizer,
            scheduler=scheduler,
            config=trainer_config,
            trial=trial
        )
        
        # 4. Delegate control to the Trainer's main `train` method.
        #    The Trainer now manages the entire stateful training loop.
        best_metric, training_log, validation_log = trainer.train()

        print("\n--- Training Workflow Complete ---")
        print(f"Best validation metric ({trainer_config.model_selection_metric}): {best_metric:.4f}")
        
        # 5. Return the results for potential further processing.
        return training_log, validation_log
    
    def run_evaluation(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes a final, rigorous evaluation of a trained model on the test set.

        This method orchestrates the evaluation workflow. It assembles the core
        components (model architecture, test data, metrics calculator) and prepares
        a unified configuration object for the dedicated Evaluator class. It then
        delegates the entire evaluation process—including loading the checkpoint,
        running sliding-window inference, and saving results—to the Evaluator.

        Args:
            checkpoint_path (Optional[str]): An explicit path to a .pth model checkpoint.
                If provided, this overrides any path specified in the config file.

        Returns:
            Dict[str, Any]: A dictionary containing the final computed metrics.
        """
        print("\n" + "="*50)
        print(">>> Starting Evaluation Workflow <<<")
        print("="*50)

        # 1. Assemble the high-level components required by the Evaluator.
        print("1/3: Assembling model, test data, and metrics calculator...")
        model = self.get_model()
        test_loader = self.get_dataloaders()['test']
        metrics_calculator = self.get_metrics_calculator()

        # 2. Prepare the unified configuration object for the Evaluator.
        #    The Evaluator class expects a single config object with attribute access.
        #    We aggregate parameters from different sections of our main YAML file.
        trainer_config = self.config.get('trainer', {})
        evaluator_config = self.config.get('evaluator', {})
        
        # --- Determine the final checkpoint path with clear priority ---
        # Priority 1: Direct argument to this method.
        # Priority 2: Path specified in the evaluator config section.
        # Priority 3: Auto-detect 'best_model.pth' in the trainer output directory.
        final_checkpoint_path = checkpoint_path or evaluator_config.get('checkpoint_path')
        if not final_checkpoint_path:
            exp_dir = trainer_config.get('output_dir')
            if not exp_dir:
                raise ValueError("Cannot auto-detect checkpoint. 'trainer.output_dir' is not set.")
            final_checkpoint_path = os.path.join(exp_dir, "best_model.pth")
            if not os.path.exists(final_checkpoint_path):
                raise FileNotFoundError(f"Could not auto-detect checkpoint at {final_checkpoint_path}")
        
        # --- Determine the final output directory ---
        eval_output_dir = evaluator_config.get('output_dir')
        if not eval_output_dir:
            eval_output_dir = os.path.join(trainer_config.get('output_dir', '.'), 'evaluation')

        eval_config_dict = {
            'DEVICE': trainer_config.get('device', 'auto'),
            'CHECKPOINT_PATH': final_checkpoint_path,
            'OUTPUT_DIR': eval_output_dir,
            'PATCH_SIZE': self.config.get('data', {}).get('patch_size', 512),
            'INFERENCE_STRIDE': trainer_config.get('inference_stride', 256),
            'INFERENCE_BATCH_SIZE': trainer_config.get('inference_batch_size', 16),
            'NUM_VISUALIZATIONS': evaluator_config.get('num_visualizations', 8),
            'PRIMARY_TASKS': self.config.get('metrics', {}).get('primary_tasks', [])
        }
        eval_config = SimpleNamespace(**eval_config_dict)

        # 3. Instantiate the dedicated Evaluator class.
        print(f"2/3: Initializing the evaluation engine...")
        evaluator = Evaluator(
            model=model,
            test_loader=test_loader,
            metrics_calculator=metrics_calculator,
            config=eval_config
        )

        # 4. Delegate control to the Evaluator's main `evaluate` method.
        print(f"3/3: Running evaluation...")
        final_metrics = evaluator.evaluate()

        print("\n--- Evaluation Workflow Complete ---")
        return final_metrics
    
    def run_hyperparameter_study(self):
        """
        Conducts a full hyperparameter optimization study using Optuna.

        This method orchestrates the entire tuning process. It reads the study
        configuration, sets up an Optuna study, and defines an objective
        function. The objective function dynamically creates a modified configuration
        for each trial and reuses the factory's own `run_training` logic to
        execute the trial, thus eliminating code duplication and ensuring consistency.
        """
        print("\n" + "="*50)
        print(">>> Starting Hyperparameter Study Workflow <<<")
        print("="*50)
        
        study_config = self.config.get('study')
        if not study_config:
            raise ValueError("The 'study' section is missing from the configuration file.")

        with open(study_config['config_path'], 'r') as f:
            search_space = yaml.safe_load(f)

        # Helper function to modify the nested config dictionary
        def _set_nested_value(d, path, value):
            keys = path.split('.')
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = value

        # --- Define the Objective Function for Optuna ---
        def objective(trial: optuna.Trial) -> float:
            # 1. Create a deep copy of the base config for this specific trial
            trial_config = copy.deepcopy(self.config)

            # 2. Modify the config with hyperparameters suggested by Optuna
            for path, details in search_space.items():
                suggest_type = details['type']
                suggest_params = details['params']
                
                if suggest_type == 'float':
                    value = trial.suggest_float(**suggest_params)
                elif suggest_type == 'int':
                    value = trial.suggest_int(**suggest_params)
                elif suggest_type == 'categorical':
                    value = trial.suggest_categorical(**suggest_params)
                else:
                    raise ValueError(f"Unsupported suggestion type '{suggest_type}'")
                
                _set_nested_value(trial_config, path, value)
            
            # 3. Create a unique output directory for the trial
            base_output_dir = trial_config['trainer']['output_dir']
            trial_output_dir = os.path.join(base_output_dir, f"trial_{trial.number}")
            trial_config['trainer']['output_dir'] = trial_output_dir
            
            print(f"\n--- Starting Trial {trial.number} ---")
            print("Params:", trial.params)

            # 4. Instantiate a NEW factory for this trial with the modified config
            # This ensures that each trial is a clean, self-contained experiment.
            # We pass the modified config dictionary directly.
            trial_factory = ExperimentFactory(config_dict=trial_config)

            # 5. Run the training and get the final validation metric
            # Pass the trial object down so the Trainer can use it for pruning.
            training_log, validation_log = trial_factory.run_training(trial=trial)
            
            metric_to_optimize = trial_config['trainer']['model_selection_metric']
            final_metric = validation_log[metric_to_optimize][-1]
            
            return final_metric

        # --- Set up and run the Optuna Study ---
        pruner_config = study_config.get('pruner', {})
        pruner = optuna.pruners.MedianPruner(**pruner_config.get('params', {})) if pruner_config.get('type') == 'MedianPruner' else None

        study = optuna.create_study(
            study_name=study_config['name'],
            storage=study_config['storage'],
            load_if_exists=True,
            direction=study_config['direction'],
            pruner=pruner
        )

        study.optimize(objective, n_trials=study_config['n_trials'])

        print("\n--- Hyperparameter Study Complete ---")
        best_trial = study.best_trial
        print(f"Best trial value ({self.config['trainer']['model_selection_metric']}): {best_trial.value:.4f}")
        print("Best hyperparameters found:")
        for key, value in best_trial.params.items():
            print(f"  - {key}: {value}")

    def generate_visualizations(
        self,
        training_log: Optional[Dict[str, Any]] = None,
        validation_log: Optional[Dict[str, Any]] = None,
        evaluation_results: Optional[Dict[str, Any]] = None
    ):
        """
        Produces all analytical plots and qualitative result images for a trained model.

        This method is highly flexible. It can generate plots from pre-existing log
        files passed as arguments, or it can orchestrate a new inference run to
        generate the data it needs for qualitative and confusion matrix plots.

        Args:
            training_log (Optional[Dict]): The training history log from `run_training`.
            validation_log (Optional[Dict]): The validation history log from `run_training`.
            evaluation_results (Optional[Dict]): The final results dict from `run_evaluation`.
                                                 Expected to contain confusion matrices.
        """
        print("\n" + "="*50)
        print(">>> Starting Visualization Workflow <<<")
        print("="*50)
        
        vis_config = self.config.get('visualizer', {})
        trainer_config = self.config.get('trainer', {})
        output_dir = trainer_config.get('output_dir', 'experiments/default_run')

        # 1. Instantiate the Visualizer engine.
        visualizer = Visualizer(
            output_dir=output_dir,
            task_info=self.task_info,
            style=vis_config.get('style', 'seaborn-v0_8-whitegrid')
        )
        print(f"Visualizations will be saved to: {output_dir}")

        # 2. Generate plots from provided training/validation logs.
        if training_log:
            print("Generating plots from training logs...")
            warmup_ratio = self.config.get('optimizer', {}).get('params', {}).get('warmup_ratio', 0.1)
            total_steps = len(training_log.get('lr', []))
            warmup_steps = int(total_steps * warmup_ratio)
            
            visualizer.plot_training_losses(training_log, filename="training_losses.png")
            visualizer.plot_learning_rate(training_log, warmup_steps, filename="learning_rate.png")
            visualizer.plot_uncertainty_weights(training_log, filename="uncertainty_weights.png")

        if validation_log:
            print("Generating plots from validation logs...")
            visualizer.plot_validation_performance(validation_log, filename="validation_performance.png")

        # 3. Generate inference-based plots (Qualitative, Confusion Matrix).
        #    This section has two paths: use pre-computed results, or generate them now.
        if evaluation_results and 'confusion_matrices' in evaluation_results:
            print("Generating confusion matrix plots from provided evaluation results...")
            confusion_matrices = evaluation_results['confusion_matrices']
        else:
            print("Generating new data for qualitative and confusion matrix plots...")
            # --- This block runs a mini-evaluation to get necessary data ---
            model = self.get_model()
            checkpoint_path = os.path.join(output_dir, "best_model.pth")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Cannot generate visualizations. Checkpoint not found at {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            
            val_loader = self.get_dataloaders()['val']
            metrics_calculator = self.get_metrics_calculator()
            metrics_calculator.reset()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            qualitative_samples = []
            num_to_collect = vis_config.get('num_qualitative_samples', 8)

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Generating visualization data"):
                    images = batch['image'].to(device)
                    is_mtl = 'masks' in batch
                    
                    if is_mtl:
                        targets = {k: v.to(device) for k, v in batch['masks'].items()}
                    else:
                        targets = batch['mask'].to(device)
                        
                    predictions = model(images)
                    metrics_calculator.update(predictions, targets)

                    if len(qualitative_samples) < num_to_collect:
                        qualitative_samples.append({
                            'image': images.cpu(),
                            'predictions': {k: v.cpu() for k, v in predictions.items()} if is_mtl else predictions.cpu(),
                            'ground_truth': {k: v.cpu() for k, v in batch['masks'].items()} if is_mtl else batch['mask'].cpu()
                        })

            confusion_matrices = metrics_calculator.confusion_matrices
            
            # --- Now generate the plots with the newly collected data ---
            self._generate_qualitative_plots(visualizer, qualitative_samples, vis_config)

        # This part runs whether the CMs were provided or generated
        self._generate_confusion_plots(visualizer, confusion_matrices)
            
        print("\n--- Visualization Workflow Complete ---")

    def _generate_qualitative_plots(self, visualizer, samples, vis_config):
        if not samples: return
        
        # Un-batch the collected samples
        images = torch.cat([s['image'] for s in samples])
        is_mtl = isinstance(samples[0]['ground_truth'], dict)

        primary_tasks = self.config.get('metrics', {}).get('primary_tasks', [])
        
        for task_name in primary_tasks:
            if is_mtl:
                gt_masks = torch.cat([s['ground_truth'][task_name] for s in samples])
                pred_logits = torch.cat([s['predictions'][task_name] for s in samples])
                pred_masks = torch.argmax(pred_logits, dim=1)
            else: # Baseline case: assume the single mask corresponds to the first primary task
                gt_masks = torch.cat([s['ground_truth'] for s in samples])
                pred_logits = torch.cat([s['predictions'] for s in samples])
                pred_masks = torch.argmax(pred_logits, dim=1)

            visualizer.plot_qualitative_results(
                images, gt_masks, pred_masks,
                task_name=task_name.capitalize(),
                filename=f"qualitative_results_{task_name}.png",
                num_samples=vis_config.get('num_qualitative_samples', 8)
            )

    def _generate_confusion_plots(self, visualizer, confusion_matrices):
        vis_config = self.config.get('visualizer', {})

        for task_name, cm_tensor in confusion_matrices.items():
            # The visualizer now gets the class names from its own task_info
            visualizer.plot_confusion_analysis(
                cm=cm_tensor.cpu().numpy(),
                task_name=task_name,  # Pass the raw task name
                filename=f"confusion_matrix_{task_name}.png",
                threshold=vis_config.get('confusion_matrix_threshold', 10),
                top_k=vis_config.get('confusion_matrix_top_k', 3)
            )