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
        
        self.base_config_dir = base_config_dir
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
        """Configure logging for the experiment."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    f'baseline_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                )
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _validate_configs(self):
        """Validate that required config files exist."""
        for name, path in self.config_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Configuration file not found: {path}. "
                    f"Please ensure {name} config exists in {self.base_config_dir}"
                )
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
        self.logger.info(f"{'='*60}")
        
        try:
            # Initialize the experiment factory
            factory = ExperimentFactory(config_path=config_path)
            
            experiment_results = {
                'name': experiment_name,
                'config_path': config_path,
                'start_time': datetime.now().isoformat(),
                'training_completed': False,
                'evaluation_completed': False,
                'training_metrics': None,
                'test_metrics': None,
                'error': None
            }
            
            # Phase 1: Training & Validation (if not skipped)
            if not skip_training:
                self.logger.info(f"🎯 Phase 1: Training & Validation for {experiment_name}")
                try:
                    factory.run_training()
                    experiment_results['training_completed'] = True
                    self.logger.info(f"✅ Training completed successfully for {experiment_name}")
                    
                except Exception as e:
                    error_msg = f"❌ Training failed for {experiment_name}: {str(e)}"
                    self.logger.error(error_msg)
                    experiment_results['error'] = error_msg
                    return experiment_results
            else:
                self.logger.info(f"⏭️  Skipping training for {experiment_name} (evaluation only)")
            
            # Phase 2: Final Testing & Evaluation
            self.logger.info(f"🔍 Phase 2: Final Testing & Evaluation for {experiment_name}")
            try:
                test_metrics = factory.run_evaluation(checkpoint_path=checkpoint_path)
                experiment_results['test_metrics'] = test_metrics
                experiment_results['evaluation_completed'] = True
                
                # Log key metrics
                self._log_key_metrics(experiment_name, test_metrics)
                self.logger.info(f"✅ Evaluation completed successfully for {experiment_name}")
                
            except Exception as e:
                error_msg = f"❌ Evaluation failed for {experiment_name}: {str(e)}"
                self.logger.error(error_msg)
                experiment_results['error'] = error_msg
                return experiment_results
            
            experiment_results['end_time'] = datetime.now().isoformat()
            
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
        
        # Global metrics (always present)
        if 'global' in metrics:
            global_metrics = metrics['global']
            self.logger.info(f"   Global mIoU: {global_metrics.get('mIoU', 'N/A'):.4f}")
            self.logger.info(f"   Global BIoU: {global_metrics.get('BIoU', 'N/A'):.4f}")
        
        # Task-specific metrics (for MTL models)
        task_metrics = [k for k in metrics.keys() if k != 'global']
        if task_metrics:
            self.logger.info("   Task-specific metrics:")
            for task in task_metrics:
                if isinstance(metrics[task], dict):
                    miou = metrics[task].get('mIoU', 'N/A')
                    biou = metrics[task].get('BIoU', 'N/A')
                    self.logger.info(f"     {task}: mIoU={miou:.4f}, BIoU={biou:.4f}")
    
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
        
        comparison_results = {
            'comparison_start_time': datetime.now().isoformat(),
            'modes_run': modes,
            'experiments': {},
            'summary': {}
        }
        
        checkpoint_paths = checkpoint_paths or {}
        
        # Run each experiment
        for mode in modes:
            if mode not in self.config_paths:
                self.logger.warning(f"⚠️  Unknown mode '{mode}' - skipping")
                continue
                
            config_path = self.config_paths[mode]
            checkpoint_path = checkpoint_paths.get(mode)
            
            experiment_result = self.run_single_experiment(
                config_path=config_path,
                experiment_name=mode,
                skip_training=skip_training,
                checkpoint_path=checkpoint_path
            )
            
            comparison_results['experiments'][mode] = experiment_result
        
        comparison_results['comparison_end_time'] = datetime.now().isoformat()
        
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
                'success': experiment['training_completed'] and experiment['evaluation_completed'],
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
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"💾 Results saved to: {results_file}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {str(e)}")


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
            modes = ['baseline', 'mtl'] if args.mode == 'both' else [args.mode]
            
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
        print("\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

