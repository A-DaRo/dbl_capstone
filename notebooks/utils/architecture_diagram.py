"""Generate architecture confrontation diagram for Baseline vs MTL variants.

This module creates detailed architectural diagrams showing the differences between:
- Baseline SegFormer (single head, standard MLP decoder)
- MTL Focused (2 primary + 5 auxiliary tasks with cross-attention)
- MTL Holistic (7 primary tasks with cross-attention)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import yaml

if TYPE_CHECKING:
    from graphviz import Digraph

try:
    from graphviz import Digraph as GraphvizDigraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    GraphvizDigraph = None


def _load_task_info(config_path: Path) -> Tuple[List[str], Dict[str, int]]:
    """Load task names and class counts from task_definitions.yaml."""
    if not config_path.exists():
        default_tasks = ["genus", "health", "fish", "human_artifacts", "substrate", "background", "biota"]
        default_counts = {t: 10 for t in default_tasks}  # Placeholder
        return default_tasks, default_counts
    
    with config_path.open("r", encoding="utf-8") as handle:
        definitions = yaml.safe_load(handle)
    
    if not isinstance(definitions, dict):
        default_tasks = ["genus", "health", "fish", "human_artifacts", "substrate", "background", "biota"]
        default_counts = {t: 10 for t in default_tasks}
        return default_tasks, default_counts
    
    task_keys = list(definitions.keys())
    class_counts = {}
    for task_key in task_keys:
        if "id2label" in definitions[task_key]:
            class_counts[task_key] = len(definitions[task_key]["id2label"])
    
    return task_keys, class_counts


def _format_task_display_name(task: str) -> str:
    """Format task name for display."""
    return task.replace("_", " ").title()


def make_architecture_confront(
    task_definitions_path: Optional[Path] = None,
):
    """
    Generate a comprehensive side-by-side architecture diagram comparing:
    - Baseline SegFormer (single head with standard MLP decoder)
    - MTL Focused (2 primary + 5 auxiliary tasks with cross-attention)
    - MTL Holistic (7 primary tasks with full cross-attention)
    
    The diagram shows the complete pipeline from input through encoder,
    decoder (with feature exchange details), and task-specific prediction heads.
    
    Args:
        task_definitions_path: Path to task_definitions.yaml for task names and class counts
        
    Returns:
        Graphviz Digraph object ready to render
    """
    if not GRAPHVIZ_AVAILABLE:
        raise ImportError(
            "Graphviz Python package not installed. "
            "Install with: pip install graphviz"
        )
    
    if task_definitions_path is None:
        task_definitions_path = Path(__file__).resolve().parents[2] / "configs" / "task_definitions.yaml"
    
    task_keys, class_counts = _load_task_info(task_definitions_path)
    
    dot = GraphvizDigraph("ArchitectureComparison", format="png")
    dot.attr(rankdir="TB", splines="ortho", bgcolor="white", pad="0.5", nodesep="0.6", ranksep="0.8")
    dot.attr("node", fontname="Arial", fontsize="11")
    dot.attr("edge", fontname="Arial", fontsize="9")
    
    # --- BASELINE SEGFORMER ---
    with dot.subgraph(name="cluster_baseline") as baseline:
        baseline.attr(
            label="<<B>Baseline SegFormer</B><BR/><FONT POINT-SIZE='10'>Single Global Head</FONT>>",
            style="rounded,filled", 
            fillcolor="#e8f4f8",
            color="#4e79a7", 
            fontsize="14",
            labeljust="l",
            margin="16"
        )
        
        # Input
        baseline.node("b_input", "Input Image\n(B, 3, H, W)", shape="box", style="rounded,filled", 
                     fillcolor="#d4e6f1", color="#2874a6", penwidth="2")
        
        # Encoder
        baseline.node("b_encoder", "SegFormer-B2 Encoder\n(Mix Transformer)", 
                     shape="box", style="rounded,filled", fillcolor="#aed6f1", color="#1f618d", penwidth="2")
        
        # Multi-scale features
        baseline.node("b_features", "Multi-scale Features\n[F₁, F₂, F₃, F₄]", 
                     shape="box", style="rounded,dashed", fillcolor="white", color="#5499c7")
        
        # Standard MLP Decoder
        baseline.node("b_decoder", "Standard MLP Decoder\n• Unify channels (C=256)\n• Upsample & concat\n• Fuse features", 
                     shape="box", style="rounded,filled", fillcolor="#85c1e9", color="#1f618d", penwidth="2")
        
        # Single prediction head
        total_classes = sum(class_counts.values()) if class_counts else 39
        baseline.node("b_head", f"Single Prediction Head\n({total_classes} classes)\n1×1 Conv", 
                     shape="box", style="rounded,filled", fillcolor="#5dade2", color="#154360", penwidth="2")
        
        # Output
        baseline.node("b_output", f"Output Logits\n(B, {total_classes}, H, W)", 
                     shape="box", style="rounded,filled", fillcolor="#3498db", color="#154360", penwidth="2")
        
        # Connections
        baseline.edge("b_input", "b_encoder")
        baseline.edge("b_encoder", "b_features")
        baseline.edge("b_features", "b_decoder")
        baseline.edge("b_decoder", "b_head")
        baseline.edge("b_head", "b_output")
    
    # --- MTL FOCUSED ---
    with dot.subgraph(name="cluster_focused") as focused:
        focused.attr(
            label="<<B>MTL Focused</B><BR/><FONT POINT-SIZE='10'>2 Primary + 5 Auxiliary Tasks</FONT>>",
            style="rounded,filled", 
            fillcolor="#fef5e7",
            color="#f28e2b", 
            fontsize="14",
            labeljust="l",
            margin="16"
        )
        
        # Input
        focused.node("f_input", "Input Image\n(B, 3, H, W)", shape="box", style="rounded,filled", 
                    fillcolor="#fdebd0", color="#ca6f1e", penwidth="2")
        
        # Encoder
        focused.node("f_encoder", "SegFormer-B2 Encoder\n(Mix Transformer)", 
                    shape="box", style="rounded,filled", fillcolor="#fad7a0", color="#af601a", penwidth="2")
        
        # Multi-scale features
        focused.node("f_features", "Multi-scale Features\n[F₁, F₂, F₃, F₄]", 
                    shape="box", style="rounded,dashed", fillcolor="white", color="#eb984e")
        
        # Hierarchical decoder with task-specific streams
        focused.node("f_decoder", "Hierarchical Decoder\n• Unify channels (C=256)\n• Task-specific MLP streams", 
                    shape="box", style="rounded,filled", fillcolor="#f8c471", color="#af601a", penwidth="2")
        
        # Feature exchange module
        focused.node("f_xchg", "<<B>Feature Exchange</B><BR/><FONT POINT-SIZE='9'>Cross-Attention Module<BR/>(dim=128)<BR/>• Q/K/V projections<BR/>• Scaled dot-product attention<BR/>• Primary tasks only</FONT>>", 
                    shape="box", style="rounded,filled", fillcolor="#f5b7b1", color="#c0392b", penwidth="2")
        
        # Primary tasks with gating
        focused.node("f_primary", "Primary Tasks (2)\n• Genus (9 cls)\n• Health (4 cls)\n\nFull MLP + Attention\n+ Gating", 
                    shape="box", style="rounded,filled", fillcolor="#f39c12", color="#d68910", penwidth="2")
        
        # Auxiliary tasks
        focused.node("f_aux", "Auxiliary Tasks (5)\n• Fish (2 cls)\n• Human Artifacts (4 cls)\n• Substrate (6 cls)\n• Background (3 cls)\n• Biota (11 cls)\n\nLightweight 1×1 Conv", 
                    shape="box", style="rounded,filled", fillcolor="#f8c471", color="#af601a", penwidth="2")
        
        # Outputs
        focused.node("f_output", "Task-Specific Outputs\nDict[task_name, Tensor]", 
                    shape="box", style="rounded,filled", fillcolor="#f39c12", color="#9c640c", penwidth="2")
        
        # Connections
        focused.edge("f_input", "f_encoder")
        focused.edge("f_encoder", "f_features")
        focused.edge("f_features", "f_decoder")
        focused.edge("f_decoder", "f_primary", xlabel="primary\nstreams", fontsize="9", color="#af601a")
        focused.edge("f_decoder", "f_aux", xlabel="aux\nstreams", fontsize="9", color="#af601a")
        focused.edge("f_primary", "f_xchg", xlabel="attend", fontsize="9", color="#c0392b")
        focused.edge("f_xchg", "f_primary", xlabel="enrich", fontsize="9", style="dashed", color="#c0392b", penwidth="2")
        focused.edge("f_primary", "f_output")
        focused.edge("f_aux", "f_output")
    
    # --- MTL HOLISTIC ---
    with dot.subgraph(name="cluster_holistic") as holistic:
        holistic.attr(
            label="<<B>MTL Holistic</B><BR/><FONT POINT-SIZE='10'>7 Primary Tasks (Full Cross-Attention)</FONT>>",
            style="rounded,filled", 
            fillcolor="#fef5f5",
            color="#e15759", 
            fontsize="14",
            labeljust="l",
            margin="16"
        )
        
        # Input
        holistic.node("h_input", "Input Image\n(B, 3, H, W)", shape="box", style="rounded,filled", 
                     fillcolor="#fadbd8", color="#a93226", penwidth="2")
        
        # Encoder
        holistic.node("h_encoder", "SegFormer-B2 Encoder\n(Mix Transformer)", 
                     shape="box", style="rounded,filled", fillcolor="#f5b7b1", color="#922b21", penwidth="2")
        
        # Multi-scale features
        holistic.node("h_features", "Multi-scale Features\n[F₁, F₂, F₃, F₄]", 
                     shape="box", style="rounded,dashed", fillcolor="white", color="#ec7063")
        
        # Hierarchical decoder
        holistic.node("h_decoder", "Hierarchical Decoder\n• Unify channels (C=256)\n• 7 Task-specific MLP streams", 
                     shape="box", style="rounded,filled", fillcolor="#ec7063", color="#922b21", penwidth="2")
        
        # Feature exchange module (more complex)
        holistic.node("h_xchg", "<<B>Feature Exchange</B><BR/><FONT POINT-SIZE='9'>Full Cross-Attention<BR/>(dim=128)<BR/>• All 7 tasks attend to each other<BR/>• 7×6=42 attention pairs<BR/>• Q/K/V projections per task<BR/>• Scaled dot-product attention</FONT>>", 
                     shape="box", style="rounded,filled", fillcolor="#f5b7b1", color="#7b241c", penwidth="2")
        
        # All primary tasks
        task_list = "\n".join([f"• {_format_task_display_name(t)} ({class_counts.get(t, '?')} cls)" 
                               for t in task_keys])
        holistic.node("h_all_tasks", f"All Primary Tasks (7)\n{task_list}\n\nAll with Full MLP + Attention + Gating", 
                     shape="box", style="rounded,filled", fillcolor="#cb4335", color="#78281f", penwidth="2", fontcolor="white")
        
        # Outputs
        holistic.node("h_output", "Task-Specific Outputs\nDict[task_name, Tensor]", 
                     shape="box", style="rounded,filled", fillcolor="#a93226", color="#641e16", penwidth="2", fontcolor="white")
        
        # Connections
        holistic.edge("h_input", "h_encoder")
        holistic.edge("h_encoder", "h_features")
        holistic.edge("h_features", "h_decoder")
        holistic.edge("h_decoder", "h_all_tasks")
        holistic.edge("h_all_tasks", "h_xchg", xlabel="all tasks\nattend", fontsize="9", color="#7b241c")
        holistic.edge("h_xchg", "h_all_tasks", xlabel="mutual\nenrichment", fontsize="9", style="dashed", color="#7b241c", penwidth="2")
        holistic.edge("h_all_tasks", "h_output")
    
    return dot


def export_architecture_diagram(
    output_base_name: str = "architecture_comparison",
    poster_dir: Optional[Path] = None,
    report_dir: Optional[Path] = None,
    task_definitions_path: Optional[Path] = None,
    formats: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Generate and export comprehensive architecture diagram to multiple formats and locations.
    
    This function creates a detailed comparison diagram showing:
    - Complete pipeline from input to output for all three models
    - Decoder architecture differences
    - Feature exchange mechanisms (cross-attention) in MTL models
    - Task-specific heads (primary vs auxiliary)
    - Class counts for each task
    
    Args:
        output_base_name: Base filename without extension (default: "architecture_comparison")
        poster_dir: Directory for poster figures (PNG, high-DPI)
        report_dir: Directory for report figures (PDF, vector)
        task_definitions_path: Path to task_definitions.yaml for dynamic task info
        formats: List of formats to generate (default: ["png", "pdf", "svg"])
        
    Returns:
        Dictionary mapping format to output path
    """
    if not GRAPHVIZ_AVAILABLE:
        raise ImportError(
            "Graphviz Python package not installed. "
            "Install with: pip install graphviz"
        )
    
    if formats is None:
        formats = ["png", "pdf", "svg"]
    
    project_root = Path(__file__).resolve().parents[2]
    
    if poster_dir is None:
        poster_dir = project_root / "latex" / "Poster_Data_shallange" / "Result-figures"
    if report_dir is None:
        report_dir = project_root / "latex" / "Methodology" / "Result-figures"
    
    poster_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    dot = make_architecture_confront(task_definitions_path)
    
    exported_paths = {}
    
    # PNG for poster (high DPI)
    if "png" in formats:
        dot.format = "png"
        dot.attr(dpi="300")  # High resolution for posters
        poster_output = poster_dir / output_base_name
        dot.render(str(poster_output), cleanup=True)
        exported_paths["png"] = poster_output.with_suffix(".png")
    
    # PDF for report (vector)
    if "pdf" in formats:
        dot.format = "pdf"
        report_output = report_dir / output_base_name
        dot.render(str(report_output), cleanup=True)
        exported_paths["pdf"] = report_output.with_suffix(".pdf")
    
    # SVG for maximum flexibility (vector, editable)
    if "svg" in formats:
        dot.format = "svg"
        svg_output = report_dir / output_base_name
        dot.render(str(svg_output), cleanup=True)
        exported_paths["svg"] = svg_output.with_suffix(".svg")
    
    return exported_paths


def make_feature_exchange_detail():
    """
    Generate a detailed diagram showing the feature exchange mechanism.
    
    This supplementary diagram zooms into the cross-attention module to show:
    - How primary task features are transformed
    - Q/K/V projection operations
    - Attention computation (scaled dot-product)
    - Gating mechanism for feature fusion
    
    Returns:
        Graphviz Digraph object ready to render
    """
    if not GRAPHVIZ_AVAILABLE:
        raise ImportError("Graphviz not available")
    
    dot = GraphvizDigraph("FeatureExchange", format="png")
    dot.attr(rankdir="TB", splines="ortho", bgcolor="white", pad="0.5", nodesep="0.5", ranksep="0.7")
    dot.attr("node", fontname="Arial", fontsize="10")
    dot.attr("edge", fontname="Arial", fontsize="9")
    
    dot.attr(label="<<B>Feature Exchange via Cross-Attention</B><BR/><FONT POINT-SIZE='10'>Mechanism for Primary Tasks</FONT>>", 
             labelloc="t", fontsize="14")
    
    # Input features from decoder (centered horizontally)
    with dot.subgraph(name="cluster_inputs") as inputs:
        inputs.attr(rank="source", style="invis")
        inputs.node("f_genus", "Genus Features\nF_genus (B,C,H,W)", shape="box", style="rounded,filled", 
                    fillcolor="#aed6f1", color="#1f618d", penwidth="2")
        inputs.node("f_health", "Health Features\nF_health (B,C,H,W)", shape="box", style="rounded,filled", 
                    fillcolor="#f5b7b1", color="#922b21", penwidth="2")

    # Pooling level with routing points (centered)
    with dot.subgraph(name="cluster_pooling_level") as pool_level:
        pool_level.attr(rank='same', style='invis')
        # Invisible points to route the 'original' connections around the QKV cluster
        pool_level.node('pass_l', shape='point', width='0')
        pool_level.node("pool", "Context Pooling\nAvgPool 4×4", shape="box", style="rounded,filled", 
                        fillcolor="#d5dbdb", color="#34495e")
        pool_level.node('pass_r', shape='point', width='0')

    # QKV projections
    with dot.subgraph(name="cluster_qkv") as qkv:
        qkv.attr(style="rounded,dashed", color="#7f8c8d", label="Q/K/V Projections (1×1 Conv)", labelloc="t", labeljust="c")
        qkv.node("q_genus", "Q_genus\n(dim=128)", shape="ellipse", fillcolor="#85c1e9", style="filled")
        qkv.node("k_genus", "K_genus\n(dim=128)", shape="ellipse", fillcolor="#85c1e9", style="filled")
        qkv.node("v_genus", "V_genus\n(dim=128)", shape="ellipse", fillcolor="#85c1e9", style="filled")
        qkv.node("q_health", "Q_health\n(dim=128)", shape="ellipse", fillcolor="#ec7063", style="filled")
        qkv.node("k_health", "K_health\n(dim=128)", shape="ellipse", fillcolor="#ec7063", style="filled")
        qkv.node("v_health", "V_health\n(dim=128)", shape="ellipse", fillcolor="#ec7063", style="filled")
    
    # Attention computation
    dot.node("attn", "Scaled Dot-Product\nAttention\n\nAttn(Q,K,V) = softmax(QKᵀ/√d)V", 
             shape="box", style="rounded,filled", fillcolor="#f9e79f", color="#9a7d0a", penwidth="2")
    
    # Gating (external) and Projection (internal) blocks aligned horizontally
    with dot.subgraph(name="cluster_gating_projection") as sg:
        sg.attr(rank='same', style="invis")
        sg.node("gate_g", "Gating\nσ(Conv1×1)", shape="diamond", fillcolor="#d7bde2", style="filled", color="#6c3483")
        sg.node("proj_g", "MLP Projection\n(dim=128→C)", shape="box", style="rounded,filled", 
                 fillcolor="#a9dfbf", color="#1e8449")
        sg.node("proj_h", "MLP Projection\n(dim=128→C)", shape="box", style="rounded,filled", 
                 fillcolor="#a9dfbf", color="#1e8449")
        sg.node("gate_h", "Gating\nσ(Conv1×1)", shape="diamond", fillcolor="#d7bde2", style="filled", color="#6c3483")

    # Final fusion
    dot.node("fuse_g", "Fusion\ngate·F_orig + (1-gate)·F_enrich", shape="box", style="rounded,filled", 
             fillcolor="#85c1e9", color="#154360", penwidth="2")
    dot.node("fuse_h", "Fusion\ngate·F_orig + (1-gate)·F_enrich", shape="box", style="rounded,filled", 
             fillcolor="#ec7063", color="#78281f", penwidth="2")
    
    # Output
    dot.node("out_g", "Enriched Genus", shape="box", style="rounded,filled", 
             fillcolor="#3498db", color="#154360", penwidth="2", fontcolor="white")
    dot.node("out_h", "Enriched Health", shape="box", style="rounded,filled", 
             fillcolor="#cb4335", color="#641e16", penwidth="2", fontcolor="white")
    
    # --- CONNECTIONS ---
    
    # Connect features to central pooling and invisible side points (symmetric for centering)
    dot.edge("f_genus", "pool")
    dot.edge("f_health", "pool")
    dot.edge("f_genus", "pass_l", style="invis", weight="0")
    dot.edge("f_health", "pass_r", style="invis", weight="0")
    
    # Simplified: connect pooling to QKV cluster directly (symmetric on both sides)
    dot.edge("pool", "k_genus", xlabel="genus\nfeatures", color="#1f618d", weight="2")
    dot.edge("pool", "k_health", xlabel="health\nfeatures", color="#922b21", weight="2")
    
    # Symmetrically routed connections to Attention
    dot.edge("q_genus", "attn", xlabel="query\ngenus", color="#1f618d")
    dot.edge("k_health", "attn", xlabel="key\nhealth", color="#922b21", constraint="false")
    dot.edge("v_health", "attn", xlabel="value\nhealth", color="#922b21")
    dot.edge("q_health", "attn", xlabel="query\nhealth", color="#922b21")
    dot.edge("k_genus", "attn", xlabel="key\ngenus", color="#1f618d", constraint="false")
    dot.edge("v_genus", "attn", xlabel="value\ngenus", color="#1f618d")
    
    # Connections from Attention to Projection
    dot.edge("attn", "proj_g", xlabel="attended\nfeatures")
    dot.edge("attn", "proj_h", xlabel="attended\nfeatures")
    
    # Original feature connections routed on the sides to gating blocks
    dot.edge("pass_l", "gate_g", xlabel="original", style="dashed", constraint="false", color="#6c3483")
    dot.edge("pass_r", "gate_h", xlabel="original", style="dashed", constraint="false", color="#6c3483")

    # Connections to Fusion blocks
    dot.edge("proj_g", "fuse_g", xlabel="enriched")
    dot.edge("gate_g", "fuse_g", xlabel="α")

    dot.edge("proj_h", "fuse_h", xlabel="enriched")
    dot.edge("gate_h", "fuse_h", xlabel="α")

    # Connections from Fusion to Output
    dot.edge("fuse_g", "out_g")
    dot.edge("fuse_h", "out_h")
    
    return dot


def export_feature_exchange_diagram(
    output_base_name: str = "feature_exchange_detail",
    output_dir: Optional[Path] = None,
    formats: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Generate and export detailed feature exchange mechanism diagram.
    
    Args:
        output_base_name: Base filename without extension
        output_dir: Directory for output (default: latex/Methodology/Result-figures)
        formats: List of formats to generate (default: ["png", "pdf"])
        
    Returns:
        Dictionary mapping format to output path
    """
    if not GRAPHVIZ_AVAILABLE:
        raise ImportError("Graphviz not available")
    
    if formats is None:
        formats = ["png", "pdf", "svg"]
    
    project_root = Path(__file__).resolve().parents[2]
    
    if output_dir is None:
        output_dir = project_root / "latex" / "Methodology" / "Result-figures"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dot = make_feature_exchange_detail()
    
    exported_paths = {}
    
    for fmt in formats:
        dot.format = fmt
        if fmt == "png":
            dot.attr(dpi="300")
        output_path = output_dir / output_base_name
        dot.render(str(output_path), cleanup=True)
        exported_paths[fmt] = output_path.with_suffix(f".{fmt}")

    return exported_paths


if __name__ == "__main__":
    if not GRAPHVIZ_AVAILABLE:
        print("[ERROR] Graphviz not available. Install with: pip install graphviz")
        print("        On Windows: pip install graphviz")
        print("        Also ensure Graphviz system binaries are installed:")
        print("        https://graphviz.org/download/")
        exit(1)
    
    print("=" * 70)
    print("Generating Enhanced Architecture Diagrams")
    print("=" * 70)
    
    # Main comparison diagram
    print("\n[1/2] Architecture Comparison Diagram")
    print("      Shows complete pipeline for all three models:")
    print("      • Baseline: Standard MLP decoder with single global head (39 classes)")
    print("      • MTL Focused: 2 primary + 5 auxiliary tasks with cross-attention")
    print("      • MTL Holistic: 7 primary tasks with full mutual cross-attention")
    print("      • Feature exchange mechanisms (Q/K/V projections, attention, gating)")
    print("      • Decoder architecture differences (MLP vs hierarchical streams)")
    
    try:
        paths = export_architecture_diagram()
        print("\n      ✓ Exported to:")
        for fmt, path in paths.items():
            print(f"          {fmt.upper():4s}: {path}")
    except Exception as e:
        print(f"\n      ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Feature exchange detail diagram
    print("\n" + "-" * 70)
    print("\n[2/2] Feature Exchange Detail Diagram")
    print("      Shows detailed cross-attention mechanism:")
    print("      • Context pooling (4×4 AvgPool)")
    print("      • Q/K/V projections (1×1 Conv, dim=128)")
    print("      • Scaled dot-product attention computation")
    print("      • MLP projection back to feature space")
    print("      • Gating mechanism for adaptive fusion")
    
    try:
        paths = export_feature_exchange_diagram()
        print("\n      ✓ Exported to:")
        for fmt, path in paths.items():
            print(f"          {fmt.upper():4s}: {path}")
    except Exception as e:
        print(f"\n      ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\n" + "=" * 70)
    print("✓ All diagrams generated successfully!")
    print("=" * 70)
