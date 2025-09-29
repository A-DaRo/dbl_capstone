"""
Parameter Counter for Coral-MTL Models

This script calculates detailed parameter counts for Coral-MTL models, breaking down counts 
between encoder (SegFormer), task-specific decoders, and shared components.
"""

import os
import sys
import yaml
import torch
from collections import defaultdict
from typing import Dict, Any, Tuple

# Add project root and src to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))  # Script is now in root
src_path = os.path.join(project_root, 'src')
sys.path.extend([project_root, src_path])

from coral_mtl.ExperimentFactory import ExperimentFactory
from coral_mtl.model.core import CoralMTLModel, BaselineSegformer


def count_parameters(model: torch.nn.Module) -> Tuple[int, Dict[str, int]]:
    """
    Count total and per-component parameters in a model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Tuple containing:
        - Total parameter count
        - Dictionary of parameter counts per component
    """
    total_params = 0
    component_params = defaultdict(int)
    
    # Iterate through named parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            
            # Categorize parameters by component
            if "encoder" in name:
                component_params["encoder"] += param_count
            elif "decoder" in name:
                if "shared" in name:
                    component_params["shared_decoder"] += param_count
                else:
                    # Find task name from parameter name
                    for task in ["genus", "health", "fish", "human_artifacts", "substrate"]:
                        if task in name:
                            component_params[f"{task}_decoder"] += param_count
                            break
            elif "attention" in name:
                component_params["attention"] += param_count
            else:
                component_params["other"] += param_count
                
    return total_params, dict(component_params)


def analyze_model_parameters(config_path: str) -> Dict[str, Any]:
    """
    Perform detailed parameter analysis of a model based on its config.
    
    Args:
        config_path: Path to model configuration YAML
        
    Returns:
        Dictionary containing parameter analysis results
    """
    # Create experiment factory and model
    factory = ExperimentFactory(config_path=config_path)
    model = factory.get_model()
    
    # Get total parameters and component breakdown
    total_params, component_breakdown = count_parameters(model)
    
    # Calculate percentages
    component_percentages = {
        k: (v / total_params) * 100 
        for k, v in component_breakdown.items()
    }
    
    # Create analysis results
    results = {
        "model_type": model.__class__.__name__,
        "total_parameters": total_params,
        "total_parameters_millions": total_params / 1e6,
        "component_breakdown": component_breakdown,
        "component_percentages": component_percentages,
        "config_path": config_path
    }
    
    return results


def print_parameter_analysis(analysis: Dict[str, Any]) -> None:
    """
    Print formatted parameter analysis results.
    
    Args:
        analysis: Dictionary containing parameter analysis
    """
    print("\n" + "="*80)
    print(f"Model Parameter Analysis for {analysis['model_type']}")
    print("="*80)
    
    print(f"\nTotal Parameters: {analysis['total_parameters']:,}")
    print(f"Total Parameters (M): {analysis['total_parameters_millions']:.2f}M")
    
    print("\nParameter Breakdown by Component:")
    print("-"*50)
    
    # Sort components by parameter count
    sorted_components = sorted(
        analysis['component_breakdown'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Print breakdown
    for component, count in sorted_components:
        percentage = analysis['component_percentages'][component]
        print(f"{component:20s}: {count:,} ({percentage:.1f}%)")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main execution function."""
    
    # Analyze both MTL and baseline models
    config_paths = [
        os.path.join(project_root, "configs/baseline_comparisons/mtl_config.yaml"),
        os.path.join(project_root, "configs/baseline_comparisons/baseline_config.yaml")
    ]
    
    for config_path in config_paths:
        try:
            analysis = analyze_model_parameters(config_path)
            print_parameter_analysis(analysis)
        except Exception as e:
            print(f"Error analyzing {config_path}: {str(e)}")


if __name__ == "__main__":
    main()