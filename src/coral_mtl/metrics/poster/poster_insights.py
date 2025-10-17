"""Advanced poster insight visualisations using real experiment metrics."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

from .real_results_generator import (
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_OUTPUT_DIR,
    ensure_output_dir,
    load_experiment_summaries,
)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


@dataclass
class ExperimentContext:
    """Lightweight container for experiment assets needed by the generators."""

    name: str
    display_name: str
    path: Path
    summary: Dict[str, object]


def discover_experiments(root_dir: Path) -> List[ExperimentContext]:
    """Wrap experiment summaries in a context with quick access helpers."""

    summaries = load_experiment_summaries(sorted(root_dir.iterdir()))
    experiments: List[ExperimentContext] = []
    for summary in summaries:
        experiments.append(
            ExperimentContext(
                name=summary["raw_name"],
                display_name=summary["display_name"],
                path=summary["path"],
                summary=summary,
            )
        )
    return experiments


def load_task_labels_from_report(summary: Dict[str, object], task: str) -> List[str]:
    """Extract the per-class label ordering from a metrics report."""

    tasks = summary.get("metrics_report", {}).get("tasks", {})
    task_data = tasks.get(task, {}).get("ungrouped", {})
    per_class = task_data.get("per_class", {})
    return list(per_class.keys())


def load_task_labels_from_config(task: str, config_path: Path) -> List[str]:
    """Fallback label loader using task definitions when history is missing."""

    with open(config_path, "r", encoding="utf-8") as handle:
        definitions = yaml.safe_load(handle)
    id2label = definitions.get(task, {}).get("id2label", {})
    ordered = [label for _, label in sorted(id2label.items(), key=lambda item: item[0])]
    return ordered


def locate_confusion_matrix_file(exp_dir: Path) -> Optional[Path]:
    """Return the confusion matrix file for a given experiment, if available."""

    for candidate in (
        exp_dir / "test_cms.jsonl",
        exp_dir / "evaluation" / "test_cms.jsonl",
    ):
        if candidate.exists():
            return candidate
    return None


def aggregate_confusion_matrix(exp_dir: Path, task: str) -> Optional[np.ndarray]:
    """Aggregate confusion matrices across all test images for the given task."""

    cm_path = locate_confusion_matrix_file(exp_dir)
    if cm_path is None:
        return None

    aggregate: Optional[np.ndarray] = None
    with open(cm_path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            task_matrix = record.get("confusion_matrices", {}).get(task)
            if task_matrix is None:
                continue
            matrix = np.asarray(task_matrix, dtype=np.float64)
            if aggregate is None:
                aggregate = matrix
            else:
                aggregate += matrix
    return aggregate


def load_advanced_metrics(exp_dir: Path) -> Dict[str, List[float]]:
    """Load advanced per-image metrics (ASSD, HD95, ARI, VI) for an experiment."""

    metrics: Dict[str, List[float]] = defaultdict(list)
    metrics_path = exp_dir / "advanced_metrics.jsonl"
    if not metrics_path.exists():
        metrics_path = exp_dir / "evaluation" / "advanced_metrics.jsonl"
    if not metrics_path.exists():
        return metrics

    with open(metrics_path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            for key in ("ASSD", "HD95", "ARI", "VI"):
                value = record.get(key)
                if value is None:
                    continue
                try:
                    metrics[key].append(float(value))
                except (TypeError, ValueError):
                    continue
    return metrics


def load_per_image_boundary_metrics(exp_dir: Path, task: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-image tuples of (mIoU, BIoU) for the specified task."""

    cms_path = locate_confusion_matrix_file(exp_dir)
    if cms_path is None:
        return np.array([]), np.array([])

    miou_list: List[float] = []
    biou_list: List[float] = []
    with open(cms_path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            per_image = record.get("per_image_metrics", {}).get(task, {})
            try:
                miou_list.append(float(per_image.get("mIoU")))
                biou_list.append(float(per_image.get("BIoU")))
            except (TypeError, ValueError):
                continue
    return np.array(miou_list, dtype=float), np.array(biou_list, dtype=float)


class PosterInsightGenerator:
    """Generate advanced visualisations for poster-ready insights."""

    def __init__(self, experiment_root: Path = DEFAULT_EXPERIMENT_ROOT, output_dir: Path = DEFAULT_OUTPUT_DIR):
        self.experiment_root = experiment_root
        self.output_dir = ensure_output_dir(output_dir)
        self.experiments = discover_experiments(self.experiment_root)
        self.task_definition_path = Path(__file__).resolve().parents[5] / "configs" / "task_definitions.yaml"

    # ------------------------------------------------------------------
    # Figure 1: Coral class-wise misclassification rate across models
    # ------------------------------------------------------------------
    def generate_failure_modes_radar(self, task: str = "genus") -> Optional[Path]:
        """
        Generate a radar chart showing per-class misclassification rates for coral classes.
        
        Metric Description:
        For each coral class, this shows the proportion of pixels from that class that were
        misclassified (predicted as any other class). A higher value indicates the model
        struggles more with that particular coral type. This reveals which coral classes
        are most challenging for each model to correctly identify.
        
        Why both genus and health?
        - Genus: Shows morphological confusion (e.g., branching vs acropora vs pocillopora)
        - Health: Shows physiological state confusion (e.g., alive vs bleached vs dead)
        These are complementary views of model performance on coral classification.
        
        Why only coral classes?
        - Excludes "unlabeled" background to focus on actual coral identification performance
        - For genus: All classes are coral genera (some with "coral" in name, others by genus name)
        - For health: All non-background classes represent coral health states
        
        Args:
            task: Task to analyze - "genus" or "health"
        
        Returns:
            Path to saved figure, or None if insufficient data
        """
        if not self.experiments:
            return None

        # Load label reference - use ungrouped to show all coral classes
        label_reference = load_task_labels_from_report(self.experiments[0].summary, task)
        if not label_reference:
            label_reference = load_task_labels_from_config(task, self.task_definition_path)

        # Filter to coral-only classes
        # Both genus and health: exclude only "unlabeled" (all others are coral-related)
        coral_classes: List[Tuple[int, str]] = []
        for idx, label in enumerate(label_reference):
            label_lower = label.lower()
            # Exclude unlabeled background - all other classes are coral-related
            if "unlabeled" in label_lower or not label.strip():
                continue
            coral_classes.append((idx, label))
        
        if not coral_classes:
            return None

        # Calculate per-class misclassification rate for each model
        model_misclass_rates: Dict[str, List[float]] = {}
        
        for context in self.experiments:
            matrix = aggregate_confusion_matrix(context.path, task)
            if matrix is None:
                continue
            
            misclass_rates = []
            for class_idx, class_label in coral_classes:
                if class_idx >= matrix.shape[0]:
                    misclass_rates.append(0.0)
                    continue
                
                # Total pixels of this class
                total_pixels = matrix[class_idx, :].sum()
                if total_pixels == 0:
                    misclass_rates.append(0.0)
                    continue
                
                # Correctly classified pixels
                correct_pixels = matrix[class_idx, class_idx]
                
                # Misclassification rate = (total - correct) / total
                misclass_rate = (total_pixels - correct_pixels) / total_pixels
                misclass_rates.append(float(misclass_rate))
            
            model_misclass_rates[context.display_name] = misclass_rates

        if not model_misclass_rates:
            return None

        # Create radar chart
        class_labels = [label for _, label in coral_classes]
        num_classes = len(class_labels)
        angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(16, 14))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Plot each model
        colors = sns.color_palette("husl", len(model_misclass_rates))
        for idx, (model_name, rates) in enumerate(model_misclass_rates.items()):
            rates_closed = rates + [rates[0]]  # Close the polygon
            ax.plot(angles, rates_closed, label=model_name, linewidth=3.5, 
                   color=colors[idx], marker='o', markersize=8)
            ax.fill(angles, rates_closed, alpha=0.15, color=colors[idx])

        # Configure axes with larger fonts
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(class_labels, size=13 if num_classes > 10 else 14, weight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['20%', '40%', '60%', '80%'], size=14, weight='bold')
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.2)
        
        # Task-specific titles with larger fonts
        task_titles = {
            "genus": ("Coral Genus Misclassification Rate by Model", 
                     "(Morphological Confusion: Which Coral Types Are Misidentified)"),
            "health": ("Coral Health State Misclassification Rate by Model",
                      "(Physiological Confusion: Alive vs Bleached vs Dead)")
        }
        title_main, title_sub = task_titles.get(task, (f"{task.capitalize()} Misclassification Rate", ""))
        ax.set_title(f"{title_main}\n{title_sub}", pad=30, fontsize=18, weight='bold')
        
        # Legend with larger font
        ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.05), 
                 fontsize=14, frameon=True, fancybox=True, shadow=True)

        output_path = self.output_dir / f"coral_misclassification_radar_{task}.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Figure 2: Loss contribution delta stack for MTL model
    # ------------------------------------------------------------------
    def generate_loss_contribution_stack(self, model_filter: str = "mtl") -> Optional[Path]:
        target_context = None
        for context in self.experiments:
            if model_filter.lower() in context.name.lower():
                target_context = context
                break
        if target_context is None:
            return None

        history: Dict[str, Sequence[float]] = target_context.summary.get("history", {})  # type: ignore[assignment]
        loss_keys = [
            "train_weighted_genus_loss",
            "train_weighted_health_loss",
            "train_weighted_fish_loss",
            "train_weighted_human_artifacts_loss",
            "train_weighted_substrate_loss",
        ]
        loss_matrix: List[np.ndarray] = []
        labels: List[str] = []
        for key in loss_keys:
            values = history.get(key)
            if not values:
                continue
            array = np.asarray(values, dtype=float)
            if np.all(np.isnan(array)):
                continue
            loss_matrix.append(array)
            labels.append(key.replace("train_weighted_", "").replace("_loss", ""))

        if not loss_matrix:
            return None

        data = np.vstack(loss_matrix)
        epochs = np.arange(1, data.shape[1] + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.stackplot(epochs, data, labels=labels, alpha=0.75)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted Loss Contribution")
        ax.set_title(f"MTL Loss Composition Over Time – {target_context.display_name}")

        start_slice = data[:, : max(1, data.shape[1] // 5)]
        end_slice = data[:, -max(1, data.shape[1] // 5):]
        start_means = start_slice.mean(axis=1)
        end_means = end_slice.mean(axis=1)
        deltas = end_means - start_means

        annotation_lines = ["Δ (final − early):"]
        for label, delta in zip(labels, deltas):
            annotation_lines.append(f"{label}: {delta:+.3f}")
        ax.text(
            1.02,
            0.5,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        ax.legend(loc="upper right")
        output_path = self.output_dir / "loss_contribution_stack.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Figure 3: Advanced metric lollipop comparison
    # ------------------------------------------------------------------
    def generate_advanced_metric_lollipop(self) -> Optional[Path]:
        if not self.experiments:
            return None

        metric_names = ["ASSD", "HD95", "VI", "ARI"]
        statistics: Dict[str, Dict[str, Tuple[float, float]]] = {}

        for context in self.experiments:
            metrics = load_advanced_metrics(context.path)
            if not metrics:
                continue
            stats_for_model: Dict[str, Tuple[float, float]] = {}
            for metric in metric_names:
                values = metrics.get(metric)
                if not values:
                    continue
                arr = np.asarray(values, dtype=float)
                if arr.size == 0:
                    continue
                median = float(np.nanmedian(arr))
                percentile_75 = float(np.nanpercentile(arr, 75))
                stats_for_model[metric] = (median, percentile_75)
            if stats_for_model:
                statistics[context.display_name] = stats_for_model

        if not statistics:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        y_positions = np.arange(len(metric_names))
        y_offset = -0.15
        for idx, (model_name, metric_stats) in enumerate(statistics.items()):
            x_values = []
            heights = []
            for metric in metric_names:
                stat_pair = metric_stats.get(metric)
                if stat_pair is None:
                    x_values.append(np.nan)
                    heights.append(np.nan)
                    continue
                median, percentile_75 = stat_pair
                x_values.append(median)
                heights.append(percentile_75)
            offset_positions = y_positions + y_offset * (idx - (len(statistics) - 1) / 2)
            ax.hlines(offset_positions, 0, heights, color="gray", alpha=0.4)
            ax.plot(heights, offset_positions, "o", color="gray", alpha=0.4)
            ax.plot(x_values, offset_positions, "o-", label=model_name, linewidth=2)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(metric_names)
        ax.set_xlabel("Metric Value (median • 75th percentile)")
        ax.set_title("Advanced Metric Distribution – Median vs 75th Percentile")
        ax.legend()

        output_path = self.output_dir / "advanced_metric_lollipop.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Figure 4: Boundary failure density map
    # ------------------------------------------------------------------
    def generate_boundary_failure_heatmap(self, task: str = "genus") -> Optional[Path]:
        if not self.experiments:
            return None

        fig, axes = plt.subplots(len(self.experiments), 1, figsize=(8, 4 * len(self.experiments)), sharex=True)
        if len(self.experiments) == 1:
            axes = [axes]

        for ax, context in zip(axes, self.experiments):
            mious, bious = load_per_image_boundary_metrics(context.path, task)
            if mious.size == 0 or bious.size == 0:
                ax.set_visible(False)
                continue
            diff = mious - bious
            scatter = ax.scatter(mious, diff, alpha=0.3, s=15, label="Per-image")
            ax.axhline(0, color="black", linewidth=1, linestyle="--")
            mean_gap = float(np.mean(diff))
            ax.axhline(mean_gap, color="red", linestyle="-", linewidth=1.5, label=f"Mean gap {mean_gap:.3f}")
            ax.set_ylabel("mIoU − BIoU")
            ax.set_title(f"Boundary Degradation – {context.display_name}")
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("Per-image mIoU")

        output_path = self.output_dir / "boundary_failure_gap.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Figure 5: Simple pipeline block diagram
    # ------------------------------------------------------------------
    def generate_pipeline_diagram(self) -> Path:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")

        stages = [
            "CoralScapes Tiles",
            "SegFormer Backbone",
            "MTL Decoder",
            "Task Heads\n(genus, health, aux)",
            "Metrics & Diagnostics",
        ]

        x_positions = np.linspace(0.1, 0.9, len(stages))
        y_position = 0.5

        for idx, (stage, x) in enumerate(zip(stages, x_positions)):
            ax.add_patch(plt.Rectangle((x - 0.08, y_position - 0.12), 0.16, 0.24, fill=True, color=sns.color_palette()[idx % len(sns.color_palette())], alpha=0.7))
            ax.text(x, y_position, stage, ha="center", va="center", fontsize=10, color="black", weight="bold")
            if idx < len(stages) - 1:
                ax.annotate(
                    "",
                    xy=(x_positions[idx + 1] - 0.11, y_position),
                    xytext=(x + 0.09, y_position),
                    arrowprops=dict(arrowstyle="->", linewidth=2, color="black"),
                )

        ax.set_title("Hierarchical Coral Segmentation Pipeline")
        output_path = self.output_dir / "pipeline_diagram.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Entry point to produce all assets
    # ------------------------------------------------------------------
    def run_all(self) -> Dict[str, Optional[Path]]:
        outputs = {
            "coral_misclassification_radar_genus": self.generate_failure_modes_radar(task="genus"),
            "coral_misclassification_radar_health": self.generate_failure_modes_radar(task="health"),
            "loss_contribution_stack": self.generate_loss_contribution_stack(),
            "advanced_metric_lollipop": self.generate_advanced_metric_lollipop(),
            "boundary_failure_gap": self.generate_boundary_failure_heatmap(),
            "pipeline_diagram": self.generate_pipeline_diagram(),
        }
        return outputs


def main(experiment_root: Optional[str] = None, output_dir: Optional[str] = None) -> Dict[str, Optional[Path]]:
    root = Path(experiment_root) if experiment_root else DEFAULT_EXPERIMENT_ROOT
    output = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    generator = PosterInsightGenerator(root, output)
    return generator.run_all()


if __name__ == "__main__":
    results = main()
    for name, path in results.items():
        if path is None:
            print(f"[WARN] {name}: no output produced")
        else:
            print(f"[OK] {name}: {path}")