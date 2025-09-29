import os
import yaml
import math
import optuna
import copy
import torch
import torch.nn as nn
from torch.optim import Optimizer
from types import SimpleNamespace
import torch
import os
import yaml
from typing import Dict, Any, Optional, Tuple, List
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Import the model classes defined in the project structure
from .model.core import CoralMTLModel, BaselineSegformer
from .data.dataset import CoralscapesMTLDataset, CoralscapesDataset
from .data.augmentations import SegmentationAugmentation
from .engine.optimizer import create_optimizer_and_scheduler
from .engine.losses import CoralMTLLoss, CoralLoss
from .engine.loss_weighting import build_weighting_strategy
from .engine.gradient_strategies import GradientUpdateStrategy
from .metrics.metrics import AbstractCoralMetrics, CoralMTLMetrics, CoralMetrics
from .engine.trainer import Trainer
from .engine.evaluator import Evaluator
from .utils.task_splitter import MTLTaskSplitter, BaseTaskSplitter
from .metrics.metrics_storer import AdvancedMetricsProcessor, MetricsStorer 

class ExperimentFactory:
    """
    The master factory and orchestrator for all experimental workflows.
    It handles dependency injection for all major components.
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
        # Set root path for resolving relative paths in config files
        self.root_path = Path(__file__).parent.parent.parent.resolve()

        if config_dict:
            self.config = config_dict
        elif config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError("Either 'config_path' or 'config_dict' must be provided.")

        # --- Initialize caches ---
        self.task_splitter = None
        self.model = None
        self.dataloaders = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.metrics_calculator = None
        self.metrics_storer = None
        self.advanced_metrics_processor = None
        self._optimizer_scheduler = None

        # --- Resolve all relative paths in config to absolute paths ---
        self._resolve_config_paths()

        # --- Centralized Task Definition Parsing ---
        self._initialize_task_splitter()

    def _resolve_config_paths(self):
        """Resolve all relative paths in the configuration to absolute paths based on root_path."""
        print(f"--- Resolving relative paths from root: {self.root_path} ---")
        
        # Resolve data paths
        data_config = self.config.get('data', {})
        
        # Resolve dataset paths
        if 'dataset_name' in data_config:
            dataset_name = data_config['dataset_name']
            if isinstance(dataset_name, str) and (dataset_name.startswith('./') or dataset_name.startswith('../')):
                data_config['dataset_name'] = str(self.root_path / dataset_name)
                print(f"Resolved dataset_name: {dataset_name} -> {data_config['dataset_name']}")
        
        if 'data_root_path' in data_config:
            data_root_path = data_config['data_root_path']
            if isinstance(data_root_path, str) and (data_root_path.startswith('./') or data_root_path.startswith('../')):
                data_config['data_root_path'] = str(self.root_path / data_root_path)
                print(f"Resolved data_root_path: {data_root_path} -> {data_config['data_root_path']}")
        
        if 'pds_train_path' in data_config:
            pds_train_path = data_config['pds_train_path']
            if isinstance(pds_train_path, str) and (pds_train_path.startswith('./') or pds_train_path.startswith('../')):
                data_config['pds_train_path'] = str(self.root_path / pds_train_path)
                print(f"Resolved pds_train_path: {pds_train_path} -> {data_config['pds_train_path']}")
        
        if 'task_definitions_path' in data_config:
            task_def_path = data_config['task_definitions_path']
            if isinstance(task_def_path, str) and not os.path.isabs(task_def_path):
                data_config['task_definitions_path'] = str(self.root_path / task_def_path)
                print(f"Resolved task_definitions_path: {task_def_path} -> {data_config['task_definitions_path']}")
        
        # Resolve trainer output directory
        trainer_config = self.config.get('trainer', {})
        if 'output_dir' in trainer_config:
            output_dir = trainer_config['output_dir']
            if isinstance(output_dir, str) and not os.path.isabs(output_dir):
                trainer_config['output_dir'] = str(self.root_path / output_dir)
                print(f"Resolved output_dir: {output_dir} -> {trainer_config['output_dir']}")
        
        # Resolve evaluator paths
        evaluator_config = self.config.get('evaluator', {})
        if 'checkpoint_path' in evaluator_config and evaluator_config['checkpoint_path']:
            checkpoint_path = evaluator_config['checkpoint_path']
            if isinstance(checkpoint_path, str) and not os.path.isabs(checkpoint_path):
                evaluator_config['checkpoint_path'] = str(self.root_path / checkpoint_path)
                print(f"Resolved checkpoint_path: {checkpoint_path} -> {evaluator_config['checkpoint_path']}")
        
        if 'output_dir' in evaluator_config and evaluator_config['output_dir']:
            eval_output_dir = evaluator_config['output_dir']
            if isinstance(eval_output_dir, str) and not os.path.isabs(eval_output_dir):
                evaluator_config['output_dir'] = str(self.root_path / eval_output_dir)
                print(f"Resolved evaluator output_dir: {eval_output_dir} -> {evaluator_config['output_dir']}")

    def _initialize_task_splitter(self):
        """Instantiates the correct TaskSplitter based on the config."""
        if self.task_splitter: return

        task_def_path = self.config.get('data', {}).get('task_definitions_path')
        if not task_def_path:
            raise ValueError("'task_definitions_path' is required in the data config.")

        # Path should already be resolved to absolute by _resolve_config_paths
        task_def_path = Path(task_def_path)
        if not task_def_path.exists():
            raise FileNotFoundError(f"Task definitions file not found: {task_def_path}")
            
        with open(task_def_path, 'r') as f:
            task_definitions = yaml.safe_load(f)
        
        model_type = self.config.get('model', {}).get('type')
        if model_type == "CoralMTL":
            self.task_splitter = MTLTaskSplitter(task_definitions)
        elif model_type == "SegFormerBaseline":
            self.task_splitter = BaseTaskSplitter(task_definitions)
        else:
            raise ValueError(f"Unsupported model type '{model_type}' for TaskSplitter initialization.")

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

        model_config = self.config.get('model', {})
        model_type = model_config.get('type')
        params = model_config.get('params', {})
        print(f"--- Building model of type: {model_type} ---")

        if model_type == "CoralMTL":
            # For MTL, num_classes for each head comes from the splitter
            num_classes_dict = {
                task: len(details['grouped']['id2label']) if details['is_grouped'] else len(details['ungrouped']['id2label'])
                for task, details in self.task_splitter.hierarchical_definitions.items()
            }
            self.model = CoralMTLModel(
                encoder_name=params['backbone'],
                decoder_channel=params['decoder_channel'],
                num_classes=num_classes_dict,
                attention_dim=params['attention_dim'],
                primary_tasks=model_config.get('tasks', {}).get('primary'),
                aux_tasks=model_config.get('tasks', {}).get('auxiliary'),
                encoder_weights=params.get('encoder_weights', 'imagenet'),
                encoder_depth=params.get('encoder_depth', 5),
            )
        elif model_type == "SegFormerBaseline":
            # For baseline, num_classes is the size of the flattened space
            self.model = BaselineSegformer(
                encoder_name=params['backbone'],
                decoder_channel=params['decoder_channel'],
                num_classes=len(self.task_splitter.flat_id2label),
                encoder_weights=params.get('encoder_weights', 'imagenet'),
                encoder_depth=params.get('encoder_depth', 5),
            )
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
        if self.dataloaders: return self.dataloaders

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

        DatasetClass = CoralscapesMTLDataset if model_type == "CoralMTL" else CoralscapesDataset
        
        self.dataloaders = {}
        # Resolve common DataLoader kwargs with sensible defaults and fallbacks
        # Prefer batch_size_per_gpu, else fall back to batch_size, else 4
        resolved_bs = (
            data_config.get('batch_size_per_gpu', None)
            if data_config.get('batch_size_per_gpu', None) is not None
            else data_config.get('batch_size', 4)
        )
        num_workers = int(data_config.get('num_workers', 4))
        persistent_workers = bool(data_config.get('persistent_workers', num_workers > 0))
        prefetch_factor = data_config.get('prefetch_factor', None)
        drop_last_default = True  # usually beneficial for throughput on train

        for split in ['train', 'validation', 'test']:
            augs = train_augmentations if split == 'train' else None
            dataset = DatasetClass(
                splitter=self.task_splitter,
                split=split,
                augmentations=augs,
                patch_size=data_config.get('patch_size', 512),
                hf_dataset_name=data_config.get('dataset_name'),
                data_root_path=data_config.get('data_root_path'),
                pds_train_path=data_config.get('pds_train_path')
            )

            # Build DataLoader kwargs incrementally to keep compatibility across PyTorch versions
            dl_kwargs = dict(
                batch_size=resolved_bs,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=bool(data_config.get('drop_last', drop_last_default if split == 'train' else False)),
            )
            # Only set persistent_workers if we actually have workers > 0
            if num_workers > 0:
                dl_kwargs['persistent_workers'] = persistent_workers
                # prefetch_factor applies only when workers > 0
                if prefetch_factor is not None:
                    dl_kwargs['prefetch_factor'] = int(prefetch_factor)

            self.dataloaders[split] = DataLoader(dataset, **dl_kwargs)
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
        if self._optimizer_scheduler is not None:
            return self._optimizer_scheduler

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

        if optimizer_config.get('use_pcgrad_wrapper', False):
            print("--- Wrapping optimizer with PCGrad ---")
            from .engine.pcgrad import PCGrad
            optimizer = PCGrad(optimizer)

        # 5. Cache and return
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._optimizer_scheduler = (self.optimizer, self.scheduler)
        return self._optimizer_scheduler


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
        model_config = self.config.get('model', {})
        task_config = model_config.get('tasks', {})

        primary_tasks = task_config.get('primary')
        if primary_tasks is not None:
            primary_tasks = list(primary_tasks)
        aux_tasks = task_config.get('auxiliary', []) or []
        aux_tasks = list(aux_tasks)

        if loss_type == "CompositeHierarchical" and not primary_tasks:
            raise ValueError("CompositeHierarchical loss requires 'model.tasks.primary' to be defined in the config.")
        
        # 2. Instantiate based on type
        if loss_type == "CompositeHierarchical":
            num_classes_dict = {}
            for task_name, task_data in self.task_splitter.hierarchical_definitions.items():
                num_classes_dict[task_name] = len(task_data['grouped']['id2label']) if task_data['is_grouped'] else len(task_data['ungrouped']['id2label'])
            
            if not num_classes_dict:
                 raise ValueError("CoralMTLLoss requires 'task_definitions_path' to be set.")
            
            # Build weighting strategy (backward compatible: defaults to Uncertainty weighting)
            weighting_cfg = loss_config.get('weighting_strategy')
            strategy = build_weighting_strategy(weighting_cfg, primary_tasks, aux_tasks)
            loss_fn = CoralMTLLoss(
                num_classes=num_classes_dict,
                primary_tasks=primary_tasks,
                aux_tasks=aux_tasks,
                weighting_strategy=strategy,
                ignore_index=params.get('ignore_index', 0),
                hybrid_alpha=params.get('hybrid_alpha', 0.5),
                focal_gamma=params.get('focal_gamma', 2.0),
                splitter=self.task_splitter  # Inject TaskSplitter for dynamic label resolution
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
        if self.metrics_calculator: return self.metrics_calculator

        print("--- Building metrics calculator ---")
        metrics_config = self.config.get('metrics', {})
        model_type = self.config.get('model', {}).get('type')
        device = torch.device(self.config.get('trainer', {}).get('device', 'cpu'))

        shared_params = {
            'splitter': self.task_splitter,
            'storer': self.get_metrics_storer(),  # Add the storer parameter
            'device': device,
            'boundary_thickness': metrics_config.get('boundary_thickness', 2),
            'ignore_index': self.config.get('data', {}).get('ignore_index', 255),
            'use_async_storage': metrics_config.get('use_async_storage', True)  # Enable async storage by default
        }

        if model_type == "CoralMTL":
            self.metrics_calculator = CoralMTLMetrics(**shared_params)
        elif model_type == "SegFormerBaseline":
            self.metrics_calculator = CoralMetrics(**shared_params)
        
        return self.metrics_calculator

    def get_metrics_storer(self) -> MetricsStorer:
        """Builds the MetricsStorer for handling results persistence."""
        if self.metrics_storer: return self.metrics_storer
        
        print("--- Building metrics storer ---")
        output_dir = self.config.get('trainer', {}).get('output_dir', 'experiments/default_run')
        
        # Ensure output directory is absolute
        if not os.path.isabs(output_dir):
            output_dir = str(self.root_path / output_dir)
            
        self.metrics_storer = MetricsStorer(output_dir)
        return self.metrics_storer
    
    def get_advanced_metrics_processor(self) -> AdvancedMetricsProcessor:
        """Builds the AdvancedMetricsProcessor for Tier 2/3 metrics computation."""
        
        if hasattr(self, 'advanced_metrics_processor') and self.advanced_metrics_processor:
            return self.advanced_metrics_processor
        
        print("--- Building advanced metrics processor ---")
        
        # Check if metrics processor is enabled in config
        metrics_processor_config = self.config.get('metrics_processor', {})
        enabled = metrics_processor_config.get('enabled', False)
        
        if not enabled:
            print("Advanced metrics processor disabled in configuration")
            self.advanced_metrics_processor = None
            return None
        
        output_dir = self.config.get('trainer', {}).get('output_dir', 'experiments/default_run')
        
        # Ensure output directory is absolute
        if not os.path.isabs(output_dir):
            output_dir = str(self.root_path / output_dir)
        
        num_cpu_workers = metrics_processor_config.get('num_cpu_workers', 30)
        enabled_tasks = metrics_processor_config.get('tasks', ["ASSD", "HD95", "PanopticQuality", "ARI"])
        
        self.advanced_metrics_processor = AdvancedMetricsProcessor(
            output_dir=output_dir,
            num_cpu_workers=num_cpu_workers,
            enabled_tasks=enabled_tasks
        )
        
        print(f"Advanced metrics processor configured with {num_cpu_workers} workers")
        print(f"Enabled tasks: {enabled_tasks}")
        
        return self.advanced_metrics_processor


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
        print("1/5: Assembling model...")
        model = self.get_model()
        
        print("2/5: Assembling dataloaders...")
        dataloaders = self.get_dataloaders()

        
        print("3/5: Assembling optimizer, scheduler, loss, and metrics...")
        optimizer, scheduler = self.get_optimizer_and_scheduler()
        loss_fn = self.get_loss_function()
        metrics_calculator = self.get_metrics_calculator()
        metrics_storer = self.get_metrics_storer()
        metrics_processor = self.get_advanced_metrics_processor()  # Get Tier 2/3 processor

        # 2. Prepare the trainer-specific configuration object.
        # --- Prepare configuration object for the Trainer ---
        print("4/5: Preparing trainer configuration...")
        trainer_config = SimpleNamespace(**self.config.get('trainer', {}))
        device = getattr(trainer_config, 'device', 'auto')
        if device == 'auto':
            trainer_config.device = "cuda" if torch.cuda.is_available() else "cpu"
        pcgrad_cfg = self.config.get('trainer', {}).get('pcgrad', {})
        trainer_config.pcgrad = SimpleNamespace(**pcgrad_cfg) if pcgrad_cfg else SimpleNamespace(enabled=False)

        strategy_instance = getattr(loss_fn, 'weighting_strategy', None)
        if isinstance(strategy_instance, GradientUpdateStrategy):
            trainer_config.strategy_type = 'gradient'
        else:
            trainer_config.strategy_type = 'loss'
        print(f"--- Trainer strategy type: {trainer_config.strategy_type} ---")
        
        # Inject other required configs
        patch_size = self.config.get('data', {}).get('patch_size', 512)
        if isinstance(patch_size, int):
            trainer_config.patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, (list, tuple)) and len(patch_size) == 2:
            trainer_config.patch_size = tuple(patch_size)
        else:
            raise ValueError("Trainer patch_size must be int or sequence of length 2.")

        stride_value = getattr(trainer_config, 'inference_stride', trainer_config.patch_size)
        if isinstance(stride_value, int):
            trainer_config.inference_stride = (stride_value, stride_value)
        elif isinstance(stride_value, (list, tuple)) and len(stride_value) == 2:
            trainer_config.inference_stride = tuple(stride_value)
        else:
            raise ValueError("trainer.inference_stride must be int or sequence of length 2.")
        
        # 4. Move model to the correct device
        print(f"5/5: Moving model and loss to device: {trainer_config.device}")
        target_device = torch.device(trainer_config.device)
        model = model.to(target_device)
        loss_fn = loss_fn.to(target_device)
        

         # --- Instantiate and run the Trainer ---
        trainer = Trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['validation'],
            loss_fn=loss_fn,
            metrics_calculator=metrics_calculator,
            metrics_storer=metrics_storer,
            optimizer=optimizer,
            scheduler=scheduler,
            config=trainer_config,
            trial=trial,
            metrics_processor=metrics_processor  # Add Tier 2/3 processor
        )
        
        trainer.train()

    
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

        # --- Assemble components for evaluation ---
        model = self.get_model()
        test_loader = self.get_dataloaders()['test']
        metrics_calculator = self.get_metrics_calculator()
        metrics_storer = self.get_metrics_storer()
        metrics_processor = self.get_advanced_metrics_processor()  # Get Tier 2/3 processor
        loss_fn = self.get_loss_function()
        
        # --- Prepare configuration object for the Evaluator ---
        trainer_config = self.config.get('trainer', {})
        eval_config_dict = self.config.get('evaluator', {})
        
        # Determine checkpoint path with clear priority
        final_checkpoint_path = checkpoint_path or eval_config_dict.get('checkpoint_path')
        if not final_checkpoint_path:
            exp_dir = trainer_config.get('output_dir')
            if not exp_dir: raise ValueError("Cannot auto-detect checkpoint. 'trainer.output_dir' is not set.")
            final_checkpoint_path = os.path.join(exp_dir, "best_model.pth")
        
        # Ensure checkpoint path is absolute
        if not os.path.isabs(final_checkpoint_path):
            final_checkpoint_path = str(self.root_path / final_checkpoint_path)
        
        # Determine output directory
        eval_output_dir = eval_config_dict.get('output_dir') or os.path.join(trainer_config.get('output_dir', '.'), 'evaluation')
        
        # Ensure evaluation output directory is absolute
        if not os.path.isabs(eval_output_dir):
            eval_output_dir = str(self.root_path / eval_output_dir)

        eval_patch_size = self.config.get('data', {}).get('patch_size', 512)
        if isinstance(eval_patch_size, int):
            eval_patch_size = (eval_patch_size, eval_patch_size)
        elif isinstance(eval_patch_size, (list, tuple)) and len(eval_patch_size) == 2:
            eval_patch_size = tuple(eval_patch_size)
        else:
            raise ValueError("Evaluation patch_size must be int or sequence of length 2.")

        eval_stride = eval_config_dict.get('inference_stride', 256)
        if isinstance(eval_stride, int):
            eval_stride = (eval_stride, eval_stride)
        elif isinstance(eval_stride, (list, tuple)) and len(eval_stride) == 2:
            eval_stride = tuple(eval_stride)
        else:
            raise ValueError("evaluator.inference_stride must be int or sequence of length 2.")

        eval_config = SimpleNamespace(
            device=trainer_config.get('device', 'auto'),
            checkpoint_path=final_checkpoint_path,
            output_dir=eval_output_dir,
            patch_size=eval_patch_size,
            inference_stride=eval_stride,
            inference_batch_size=eval_config_dict.get('inference_batch_size', 16),
        )
        if eval_config.device == 'auto':
            eval_config.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move model to the correct device before evaluation
        print(f"Moving model to device: {eval_config.device}")
        target_device = torch.device(eval_config.device)
        model = model.to(target_device)
        if loss_fn is not None:
            loss_fn = loss_fn.to(target_device)
            loss_fn.eval()

        # --- Instantiate and run the Evaluator ---
        evaluator = Evaluator(
            model=model,
            test_loader=test_loader,
            metrics_calculator=metrics_calculator,
            metrics_storer=metrics_storer,
            config=eval_config,
            metrics_processor=metrics_processor,  # Add Tier 2/3 processor
            loss_fn=loss_fn
        )
        
        final_metrics = evaluator.evaluate()
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

        # Resolve study config path
        config_path = study_config['config_path']
        if not os.path.isabs(config_path):
            config_path = str(self.root_path / config_path)
            
        with open(config_path, 'r') as f:
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
            # Ensure base output directory is absolute
            if not os.path.isabs(base_output_dir):
                base_output_dir = str(self.root_path / base_output_dir)
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