"""Standalone smoke script for exercising baseline and MTL training pipelines.

This script mirrors the intent of ``experiments/baselines_comparison/train_val_test_script.py``
but keeps execution lightweight. For each configuration in ``tests/configs`` we run a short
training session followed by evaluation, toggling whether Poisson Disk Sampling (PDS) patches
are used for the training split. All other configuration fields remain untouched so the
ExperimentFactory wiring is validated exactly as specified.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from coral_mtl.ExperimentFactory import ExperimentFactory


TESTS_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = TESTS_ROOT / "configs"
PDS_ROOT = TESTS_ROOT / "dataset" / "processed" / "pds_patches"

CONFIG_REGISTRY: Dict[str, Path] = {
	"mtl": CONFIG_ROOT / "test_config_mtl.yaml",
	"baseline": CONFIG_ROOT / "test_config_baseline.yaml",
}


def _configure_pds(config: Dict[str, Any], use_pds: bool) -> Dict[str, Any]:
	"""Return a copy of ``config`` with only the PDS setting adjusted."""

	updated = copy.deepcopy(config)
	data_cfg = updated.get("data", {})

	if use_pds:
		if not PDS_ROOT.exists():
			raise FileNotFoundError(
				"Requested PDS patches but directory is missing: " f"{PDS_ROOT}"
			)
		data_cfg["pds_train_path"] = str(PDS_ROOT)
	else:
		data_cfg["pds_train_path"] = None

	updated["data"] = data_cfg
	return updated


def _summarize_metrics(metrics: Dict[str, Any], top_k: int = 5) -> Dict[str, float]:
	"""Extract the top-level optimization metrics (up to ``top_k`` entries)."""

	optim = metrics.get("optimization_metrics", {})
	summary = {}
	for idx, (key, value) in enumerate(optim.items()):
		if idx >= top_k:
			break
		if isinstance(value, (int, float)):
			summary[key] = float(value)
	return summary


def run_trial(config_name: str, use_pds: bool) -> Dict[str, Any]:
	"""Run training + evaluation for a single configuration and PDS toggle."""

	config_path = CONFIG_REGISTRY[config_name]
	config = yaml.safe_load(config_path.read_text())
	run_config = _configure_pds(config, use_pds=use_pds)

	factory = ExperimentFactory(config_dict=run_config)

	# Ensure previous artefacts do not leak into the new run while retaining config values
	output_dir = Path(factory.config.get("trainer", {}).get("output_dir", ""))
	if output_dir:
		shutil.rmtree(output_dir, ignore_errors=True)

	print(f"\n=== Running {config_name.upper()} | PDS={'ON' if use_pds else 'OFF'} ===")
	training_result = factory.run_training()
	best_metric: float | None = None

	if isinstance(training_result, (int, float)):
		best_metric = float(training_result)
	else:
		# Attempt to recover the best metric from the saved history
		model_metric = factory.config.get("trainer", {}).get("model_selection_metric", "global.BIoU")
		history_path = output_dir / "history.json"
		if history_path.exists():
			try:
				history_data = json.loads(history_path.read_text())
				metric_series = history_data.get(model_metric, [])
				if metric_series:
					best_metric = float(max(metric_series))
			except Exception as exc:  # pragma: no cover - best effort logging
				print(f"Warning: Failed to parse history for {model_metric}: {exc}")

	if best_metric is not None:
		print(f"Best validation metric: {best_metric:.6f}")
	else:
		print("Best validation metric unavailable (factory did not return a scalar).")

	metrics = factory.run_evaluation()
	summary = _summarize_metrics(metrics)

	print("Key optimization metrics:")
	print(json.dumps(summary, indent=2))


	return {
		"best_metric": best_metric,
		"metrics": summary,
	}


def main() -> None:
    """Entry point for the standalone smoke run."""

    parser = argparse.ArgumentParser(description="Run baseline and MTL training pipelines.")
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
    args = parser.parse_args()

    overall_results: Dict[str, Dict[str, Any]] = {}
    modes_to_run = [args.mode] if args.mode != 'both' else CONFIG_REGISTRY.keys()

    for config_name in modes_to_run:
        for use_pds in (False, True):
            result_key = f"{config_name}_{'pds' if use_pds else 'nopds'}"
            if args.skip_training:
                print(f"\n=== Skipping training for {config_name.upper()} | PDS={'ON' if use_pds else 'OFF'} ===")
                factory = ExperimentFactory(config_dict=_configure_pds(yaml.safe_load(CONFIG_REGISTRY[config_name].read_text()), use_pds=use_pds))
                metrics = factory.run_evaluation()
                summary = _summarize_metrics(metrics)
                overall_results[result_key] = {"metrics": summary}
            else:
                overall_results[result_key] = run_trial(config_name, use_pds)

    print("\n=== Summary ===")
    print(json.dumps(overall_results, indent=2))


if __name__ == "__main__":
	main()
