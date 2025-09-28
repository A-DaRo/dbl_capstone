#!/usr/bin/env python
"""
Full Train/Val/Test Script for Baseline Comparisons

This script executes comprehensive training, validation, and testing workflows
for both MTL (Multi-Task Learning) and non-MTL baseline models using the 
configurations defined in configs/baseline_comparisons/.

The script supports:
- Baseline SegFormer model (non-MTL) 
- CoralMTL model (MTL with hierarchical task structure)
- Complete train/validation/test pipeline
- Automated checkpoint management
- Comprehensive evaluation and metrics reporting
- Results comparison between approaches

Usage:
    python train_val_test_script.py [--config CONFIG_PATH] [--mode MODE] [--skip-training]

Arguments:
    --config: Path to specific config file (optional, runs both by default)
    --mode: Specific mode to run: 'baseline', 'mtl', or 'both' (default: 'both')
    --skip-training: Skip training and run evaluation only
    --eval-only: Run evaluation only (same as --skip-training)
    --checkpoint: Specific checkpoint path for evaluation
"""

import os
import sys
import argparse
import logging
import torch
import platform
import atexit
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from coral_mtl.ExperimentFactory import ExperimentFactory


class BaselineComparison:
    """
    Orchestrator class for running comprehensive baseline comparisons.
    
    This class manages the full experimental pipeline for comparing baseline
    SegFormer models with CoralMTL models, including training, validation,
    testing, and results aggregation.
    """
    
    def __init__(self, base_config_dir: str = None):
        """
        Initialize the baseline comparison orchestrator.
        
        Args:
            base_config_dir: Directory containing baseline configuration files
        """
        # Set default config directory relative to project root
        if base_config_dir is None:
            # Get the project root directory (2 levels up from this script)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            base_config_dir = os.path.join(project_root, "configs", "baseline_comparisons")
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.base_config_dir = base_config_dir
        self.script_dir = script_dir
        self.results = {}
        self.config_paths = {
            'baseline': os.path.join(base_config_dir, 'baseline_config.yaml'),
            'mtl': os.path.join(base_config_dir, 'mtl_config.yaml')
        }
        
        # Setup logging
        self._setup_logging()
        
        # Validate config files exist
        self._validate_configs()
    
    def _setup_logging(self):
        """Configure detailed logging for the experiment."""
        # Create a more detailed log format
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        
        # Create timestamp for log filename and ensure it lives next to this script as .md
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir_path = Path(self.script_dir)
        script_dir_path.mkdir(parents=True, exist_ok=True)
        log_path = script_dir_path / f'baseline_comparison_{timestamp}.md'
        self.log_filename = str(log_path)
        
        # Remove any existing handlers to avoid duplicates
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Configure logging with more detailed settings
        logging.basicConfig(
            level=logging.DEBUG,  # Set to DEBUG for maximum verbosity
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.log_filename, mode='w', encoding='utf-8')
            ],
            force=True  # Force reconfiguration
        )
        
        # Get logger for this module
        self.logger = logging.getLogger(__name__)
        
        # Also configure third-party loggers to be less verbose
        logging.getLogger('torch').setLevel(logging.WARNING)
        logging.getLogger('torchvision').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        
        # Log initial setup information
        self.logger.info(f"Logging configured with file: {self.log_filename}")
        self.logger.info(f"Log level set to: {logging.getLevelName(logging.DEBUG)}")
        self.logger.info(f"Python version: {sys.version}")
        self.logger.info(f"PyTorch version: {torch.__version__ if 'torch' in sys.modules else 'Not loaded'}")
        self.logger.info(f"Current working directory: {os.getcwd()}")
        self.logger.info(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
        
        # Ensure logs are flushed on normal interpreter shutdown
        atexit.register(logging.shutdown)

        # Install SIGINT handler to log and flush on Ctrl-C
        self._install_interrupt_handler()

        # Log system information
        import platform
        self.logger.info(f"Platform: {platform.platform()}")
        self.logger.info(f"Architecture: {platform.architecture()}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA available: True, devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                self.logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            self.logger.info("CUDA available: False")

    def _install_interrupt_handler(self):
        """Install a SIGINT handler that logs and forces a clean flush on Ctrl-C."""
        logger = logging.getLogger(__name__)

        def _sigint_handler(signum, frame):
            try:
                logger.error("Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...", exc_info=True)
                # Flush all handlers explicitly
                root = logging.getLogger()
                for h in list(root.handlers):
                    try:
                        h.flush()
                    except Exception:
                        pass
            finally:
                # Re-raise as KeyboardInterrupt to trigger normal abort flow
                raise KeyboardInterrupt()

        try:
            signal.signal(signal.SIGINT, _sigint_handler)
        except Exception:
            # Some environments may not allow installing signal handlers
            logger.debug("Skipping SIGINT handler installation (unsupported environment)")
    
    @staticmethod
    def _format_metric_value(value: Any) -> str:
        """Return a human-readable metric representation."""

        if isinstance(value, (int, float)):
            return f"{value:.4f}"
        return "N/A"

    def _validate_configs(self):
        """Validate that required config files exist."""
        self.logger.debug("Starting configuration validation")
        for name, path in self.config_paths.items():
            self.logger.debug(f"Checking {name} config at path: {path}")
            if not os.path.exists(path):
                self.logger.error(f"Configuration file not found: {path}")
                raise FileNotFoundError(
                    f"Configuration file not found: {path}. "
                    f"Please ensure {name} config exists in {self.base_config_dir}"
                )
            self.logger.debug(f"✓ {name} config file exists")
        self.logger.info("All configuration files validated successfully")
    
    def run_single_experiment(
        self, 
        config_path: str, 
        experiment_name: str,
        skip_training: bool = False,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a complete experiment (train + test) for a single configuration.
        
        Args:
            config_path: Path to the experiment configuration file
            experiment_name: Name identifier for this experiment
            skip_training: If True, skip training and run evaluation only
            checkpoint_path: Specific checkpoint to use for evaluation
            
        Returns:
            Dictionary containing experiment results and metrics
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {experiment_name.upper()} Experiment")
        self.logger.info(f"Config: {config_path}")
        self.logger.info(f"Skip training: {skip_training}")
        self.logger.info(f"Checkpoint path: {checkpoint_path if checkpoint_path else 'Auto-detect best'}")
        self.logger.info(f"{'='*60}")
        
        try:
            # Initialize the experiment factory
            self.logger.debug(f"Initializing ExperimentFactory with config: {config_path}")
            factory = ExperimentFactory(config_path=config_path)
            self.logger.debug("ExperimentFactory initialized successfully")
            
            experiment_results = {
                'name': experiment_name,
                'config_path': config_path,
                'start_time': datetime.now().isoformat(),
                'training_completed': False,
                'evaluation_completed': False,
                'training_metrics': None,
                'test_metrics': None,
                'error': None,
                'skip_training': skip_training
            }
            
            # Log experiment configuration details
            self.logger.debug("Logging experiment configuration details...")
            try:
                config_dict = getattr(factory, 'config', {})
                if isinstance(config_dict, dict):
                    model_cfg = config_dict.get('model', {})
                    trainer_cfg = config_dict.get('trainer', {})
                    optimizer_cfg = config_dict.get('optimizer', {})

                    self.logger.info(f"Model type: {model_cfg.get('type', 'Unknown')}")
                    self.logger.info(f"Primary tasks: {model_cfg.get('tasks', {}).get('primary', [])}")
                    self.logger.info(f"Auxiliary tasks: {model_cfg.get('tasks', {}).get('auxiliary', [])}")
                    self.logger.info(f"Device policy: {trainer_cfg.get('device', 'auto')}")
                    self.logger.info(f"Batch size per GPU: {config_dict.get('data', {}).get('batch_size_per_gpu', 'Unknown')}")
                    self.logger.info(f"Learning rate: {optimizer_cfg.get('params', {}).get('lr', 'Unknown')}")
                    self.logger.info(f"Epochs: {trainer_cfg.get('epochs', 'Unknown')}")
                    self.logger.info(f"Output directory: {trainer_cfg.get('output_dir', 'Unknown')}")
            except Exception as e:
                self.logger.warning(f"Could not log config details: {str(e)}")
            
            # Phase 1: Training & Validation (if not skipped)
            if not skip_training:
                self.logger.info(f"🎯 Phase 1: Training & Validation for {experiment_name}")
                self.logger.debug("Starting training phase...")
                try:
                    start_time = datetime.now()
                    self.logger.debug(f"Training started at: {start_time}")
                    
                    factory.run_training()
                    
                    end_time = datetime.now()
                    training_duration = end_time - start_time
                    self.logger.info(f"Training duration: {training_duration}")
                    
                    experiment_results['training_completed'] = True
                    self.logger.info(f"✅ Training completed successfully for {experiment_name}")
                    
                except Exception as e:
                    error_msg = f"❌ Training failed for {experiment_name}: {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    experiment_results['error'] = error_msg
                    return experiment_results
            else:
                self.logger.info(f"⏭️  Skipping training for {experiment_name} (evaluation only)")
                experiment_results['training_completed'] = True
            
            # Phase 2: Final Testing & Evaluation
            self.logger.info(f"🔍 Phase 2: Final Testing & Evaluation for {experiment_name}")
            self.logger.debug("Starting evaluation phase...")
            try:
                eval_start_time = datetime.now()
                self.logger.debug(f"Evaluation started at: {eval_start_time}")
                self.logger.debug(f"Using checkpoint: {checkpoint_path if checkpoint_path else 'Auto-detect best'}")
                
                test_metrics = factory.run_evaluation(checkpoint_path=checkpoint_path)
                
                eval_end_time = datetime.now()
                eval_duration = eval_end_time - eval_start_time
                self.logger.info(f"Evaluation duration: {eval_duration}")
                
                experiment_results['test_metrics'] = test_metrics
                experiment_results['evaluation_completed'] = True
                
                # Log key metrics
                self._log_key_metrics(experiment_name, test_metrics)
                self.logger.info(f"✅ Evaluation completed successfully for {experiment_name}")
                
            except Exception as e:
                error_msg = f"❌ Evaluation failed for {experiment_name}: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                experiment_results['error'] = error_msg
                return experiment_results
            
            experiment_results['end_time'] = datetime.now().isoformat()
            total_duration = datetime.fromisoformat(experiment_results['end_time']) - datetime.fromisoformat(experiment_results['start_time'])
            self.logger.info(f"Total experiment duration: {total_duration}")
            
            # Log completion
            self.logger.info(f"🎉 {experiment_name.upper()} experiment completed successfully!")
            
            return experiment_results
            
        except Exception as e:
            error_msg = f"❌ Unexpected error in {experiment_name} experiment: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'name': experiment_name,
                'config_path': config_path,
                'error': error_msg,
                'training_completed': False,
                'evaluation_completed': False
            }
    
    def _log_key_metrics(self, experiment_name: str, metrics: Dict[str, Any]):
        """Log key performance metrics in a readable format."""
        self.logger.info(f"📊 Key Metrics for {experiment_name.upper()}:")
        self.logger.debug(f"Full metrics structure for {experiment_name}: {json.dumps(metrics, indent=2, default=str)}")
        
        # Global metrics (always present)
        if 'global' in metrics:
            global_metrics = metrics['global']
            self.logger.info(
                f"   Global mIoU: {self._format_metric_value(global_metrics.get('mIoU'))}"
            )
            self.logger.info(
                f"   Global BIoU: {self._format_metric_value(global_metrics.get('BIoU'))}"
            )
            self.logger.debug(f"   Global pixel accuracy: {global_metrics.get('pixel_accuracy', 'N/A')}")
            if 'TIDE_errors' in global_metrics and global_metrics['TIDE_errors']:
                tide = global_metrics['TIDE_errors']
                self.logger.debug(
                    f"   Global TIDE - Classification error: {self._format_metric_value(tide.get('classification_error'))}"
                )
                self.logger.debug(
                    f"   Global TIDE - Background error: {self._format_metric_value(tide.get('background_error'))}"
                )
                self.logger.debug(
                    f"   Global TIDE - Missed error: {self._format_metric_value(tide.get('missed_error'))}"
                )
        
        # Task-specific metrics (for MTL models)
        task_metrics = [k for k in metrics.keys() if k != 'global']
        if task_metrics:
            self.logger.info("   Task-specific metrics:")
            for task in task_metrics:
                if isinstance(metrics[task], dict):
                    miou = self._format_metric_value(metrics[task].get('mIoU'))
                    biou = self._format_metric_value(metrics[task].get('BIoU'))
                    self.logger.info(f"     {task}: mIoU={miou}, BIoU={biou}")
                    self.logger.debug(f"     {task} detailed metrics: {json.dumps(metrics[task], indent=2, default=str)}")
        
        # Log optimization metrics if available
        if 'optimization_metrics' in metrics:
            self.logger.debug("Optimization metrics summary:")
            for key, value in metrics['optimization_metrics'].items():
                self.logger.debug(f"   {key}: {value}")
    
    def run_comparison(
        self, 
        modes: List[str] = ['baseline', 'mtl'],
        skip_training: bool = False,
        checkpoint_paths: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Run complete baseline comparison across specified modes.
        
        Args:
            modes: List of modes to run ('baseline', 'mtl', or both)
            skip_training: Skip training phase and run evaluation only
            checkpoint_paths: Optional dict mapping mode names to checkpoint paths
            
        Returns:
            Dictionary containing all experiment results
        """
        self.logger.info("🔬 Starting Comprehensive Baseline Comparison")
        self.logger.info(f"Modes to run: {', '.join(modes)}")
        self.logger.info(f"Skip training: {skip_training}")
        self.logger.debug(f"Config directory: {self.base_config_dir}")
        self.logger.debug(f"Available config files: {list(self.config_paths.keys())}")
        
        comparison_results = {
            'comparison_start_time': datetime.now().isoformat(),
            'modes_run': modes,
            'experiments': {},
            'summary': {}
        }
        
        checkpoint_paths = checkpoint_paths or {}
        self.logger.debug(f"Checkpoint paths provided: {checkpoint_paths}")
        
        # Run each experiment
        for i, mode in enumerate(modes, 1):
            self.logger.info(f"\n🚀 Running experiment {i}/{len(modes)}: {mode}")
            
            if mode not in self.config_paths:
                self.logger.warning(f"⚠️  Unknown mode '{mode}' - skipping")
                continue
                
            config_path = self.config_paths[mode]
            checkpoint_path = checkpoint_paths.get(mode)
            
            self.logger.debug(f"Config path for {mode}: {config_path}")
            self.logger.debug(f"Checkpoint path for {mode}: {checkpoint_path}")
            
            experiment_result = self.run_single_experiment(
                config_path=config_path,
                experiment_name=mode,
                skip_training=skip_training,
                checkpoint_path=checkpoint_path
            )
            
            comparison_results['experiments'][mode] = experiment_result
            
            # Log intermediate results
            if experiment_result.get('error'):
                self.logger.error(f"❌ Experiment {mode} failed: {experiment_result['error']}")
            else:
                self.logger.info(f"✅ Experiment {mode} completed successfully")
        
        comparison_results['comparison_end_time'] = datetime.now().isoformat()
        
        # Calculate total comparison duration
        start_dt = datetime.fromisoformat(comparison_results['comparison_start_time'])
        end_dt = datetime.fromisoformat(comparison_results['comparison_end_time'])
        total_duration = end_dt - start_dt
        self.logger.info(f"Total comparison duration: {total_duration}")
        
        # Generate summary
        self._generate_comparison_summary(comparison_results)
        
        # Save results
        self._save_results(comparison_results)
        
        return comparison_results
    
    def _generate_comparison_summary(self, results: Dict[str, Any]):
        """Generate a summary comparing results across experiments."""
        self.logger.info("\n" + "="*60)
        self.logger.info("📈 BASELINE COMPARISON SUMMARY")
        self.logger.info("="*60)
        
        summary = {}
        
        for mode, experiment in results['experiments'].items():
            summary[mode] = {
                'success': (
                    experiment.get('evaluation_completed', False)
                    and (experiment.get('training_completed', False) or experiment.get('skip_training', False))
                ),
                'error': experiment.get('error'),
                'key_metrics': {}
            }
            
            if experiment['evaluation_completed'] and experiment['test_metrics']:
                metrics = experiment['test_metrics']
                
                # Extract key metrics for comparison
                if 'global' in metrics:
                    summary[mode]['key_metrics']['global_miou'] = metrics['global'].get('mIoU', 0)
                    summary[mode]['key_metrics']['global_biou'] = metrics['global'].get('BIoU', 0)
        
        results['summary'] = summary
        
        # Log comparison
        successful_experiments = [mode for mode, data in summary.items() if data['success']]
        
        if len(successful_experiments) > 1:
            self.logger.info("🏆 Performance Comparison:")
            
            # Compare global mIoU
            miou_scores = {
                mode: summary[mode]['key_metrics'].get('global_miou', 0) 
                for mode in successful_experiments
            }
            best_miou_mode = max(miou_scores.keys(), key=lambda k: miou_scores[k])
            
            for mode in successful_experiments:
                miou = miou_scores[mode]
                indicator = "🥇" if mode == best_miou_mode else "  "
                self.logger.info(f"   {indicator} {mode.upper()}: Global mIoU = {miou:.4f}")
        
        # Log any failures
        failed_experiments = [mode for mode, data in summary.items() if not data['success']]
        if failed_experiments:
            self.logger.warning("❌ Failed experiments:")
            for mode in failed_experiments:
                error = summary[mode]['error'] or "Unknown error"
                self.logger.warning(f"   - {mode.upper()}: {error}")
        
        self.logger.info("="*60)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"baseline_comparison_results_{timestamp}.json"
        
        try:
            # Save with pretty formatting
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            
            # Also save a summary version for quick reference
            summary_file = f"baseline_comparison_summary_{timestamp}.json"
            summary_data = {
                'timestamp': timestamp,
                'modes_run': results.get('modes_run', []),
                'total_duration': results.get('comparison_end_time', '') and results.get('comparison_start_time', ''),
                'summary': results.get('summary', {}),
                'key_metrics': {}
            }
            
            # Extract key metrics for quick reference
            for mode, experiment in results.get('experiments', {}).items():
                if experiment.get('test_metrics') and 'optimization_metrics' in experiment['test_metrics']:
                    opt_metrics = experiment['test_metrics']['optimization_metrics']
                    summary_data['key_metrics'][mode] = {
                        'global_miou': opt_metrics.get('global.mIoU', 'N/A'),
                        'global_biou': opt_metrics.get('global.BIoU', 'N/A')
                    }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"💾 Full results saved to: {results_file}")
            self.logger.info(f"💾 Summary results saved to: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {str(e)}", exc_info=True)


def main():
    """Main entry point for the baseline comparison script."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive baseline comparison experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to specific config file (runs both baseline and MTL by default)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['baseline', 'mtl', 'both'],
        default='both',
        help='Specific mode to run (default: both)'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training and run evaluation only'
    )
    
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Run evaluation only (same as --skip-training)'
    )
    
    parser.add_argument(
        '--checkpoint-baseline',
        type=str,
        help='Specific checkpoint path for baseline model evaluation'
    )
    
    parser.add_argument(
        '--checkpoint-mtl',
        type=str,
        help='Specific checkpoint path for MTL model evaluation'
    )
    
    parser.add_argument(
        '--config-dir',
        type=str,
        default=None,
        help='Directory containing baseline configuration files (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Handle eval-only flag
    skip_training = args.skip_training or args.eval_only
    
    try:
        # Initialize the comparison orchestrator
        comparison = BaselineComparison(base_config_dir=args.config_dir)
        
        # Determine modes to run
        if args.config:
            # Single config mode
            config_path = args.config
            experiment_name = os.path.splitext(os.path.basename(config_path))[0]
            
            checkpoint_path = None
            if 'baseline' in experiment_name.lower():
                checkpoint_path = args.checkpoint_baseline
            elif 'mtl' in experiment_name.lower():
                checkpoint_path = args.checkpoint_mtl
            
            result = comparison.run_single_experiment(
                config_path=config_path,
                experiment_name=experiment_name,
                skip_training=skip_training,
                checkpoint_path=checkpoint_path
            )
            
            print(f"\n🎯 Single experiment completed: {experiment_name}")
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
                sys.exit(1)
        else:
            # Multiple configs mode
            modes = ['mtl', 'baseline'] if args.mode == 'both' else [args.mode]
            
            checkpoint_paths = {}
            if args.checkpoint_baseline:
                checkpoint_paths['baseline'] = args.checkpoint_baseline
            if args.checkpoint_mtl:
                checkpoint_paths['mtl'] = args.checkpoint_mtl
            
            results = comparison.run_comparison(
                modes=modes,
                skip_training=skip_training,
                checkpoint_paths=checkpoint_paths if checkpoint_paths else None
            )
            
            # Check for any failures
            failed_experiments = [
                mode for mode, exp in results['experiments'].items() 
                if exp.get('error') or not (exp['training_completed'] or skip_training)
            ]
            
            if failed_experiments:
                print(f"\n❌ Some experiments failed: {', '.join(failed_experiments)}")
                sys.exit(1)
            else:
                print(f"\n🎉 All experiments completed successfully!")
        
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.error("Experiment interrupted by user (Ctrl-C)", exc_info=True)
        # Ensure all logs are flushed before exiting
        logging.shutdown()
        sys.exit(130)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        logging.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()

