"""
analysis.py - cached run analysis, summary tables, and figure rebuild entry points.

The heavy plotting/statistics helpers still live in pipeline.py for now, but the
public API for "load generated/scored CSVs and rebuild analysis outputs" lives
here. That gives figure iteration a stable Python function instead of making the
shell scripts know pipeline internals.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", Path(__file__).resolve().parent / "cache" / "matplotlib"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from chorus.pipeline import Config, PocketSpec


def scored_candidates_path(run_dir: Path, scorer: str) -> Path:
    scorer = scorer.strip().lower()
    return (
        run_dir / "scored_candidates.csv"
        if scorer == "boltz"
        else run_dir / f"scored_candidates_{scorer}.csv"
    )


def load_cached_run_tables(
    run_dir: Path,
    scorer: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list["PocketSpec"]] | None:
    """Load cached generated/unique/scored tables if a run is analysis-ready."""
    from chorus.pipeline import read_cached_pocket_specs

    generated_path = run_dir / "generated_by_generator.csv"
    unique_path = run_dir / "unique_generated.csv"
    scored_path = scored_candidates_path(run_dir, scorer)
    if not (generated_path.exists() and unique_path.exists() and scored_path.exists()):
        return None

    print(
        f"\nCached run detected: loading {generated_path.name}, "
        f"{unique_path.name}, and {scored_path.name}",
        flush=True,
    )
    generated_df = pd.read_csv(generated_path)
    unique_df = pd.read_csv(unique_path)
    scored_df = pd.read_csv(scored_path)
    pocket_specs = read_cached_pocket_specs(run_dir)
    return generated_df, unique_df, scored_df, pocket_specs


def rebuild_cached_run(
    run_dir: Path,
    cfg: "Config",
    *,
    run_name: str | None = None,
    scorer: str | None = None,
    generator_errors: dict[str, str] | None = None,
) -> Path:
    """Load a completed run from CSV caches and rebuild all analysis outputs."""
    run_name = run_name or run_dir.name
    scorer = (scorer or cfg.scorer).strip().lower()
    cached = load_cached_run_tables(run_dir, scorer)
    if cached is None:
        raise FileNotFoundError(
            "Cached run is missing generated_by_generator.csv, "
            "unique_generated.csv, or the scorer-specific scored_candidates CSV."
        )
    generated_df, _unique_df, scored_df, pocket_specs = cached
    return rebuild_analysis_outputs(
        run_dir=run_dir,
        run_name=run_name,
        cfg=cfg,
        generated_df=generated_df,
        scored_df=scored_df,
        pocket_specs=pocket_specs,
        generator_errors=generator_errors or {},
        scorer=scorer,
    )


def rebuild_analysis_outputs(
    run_dir: Path,
    run_name: str,
    cfg: "Config",
    generated_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    pocket_specs: list["PocketSpec"],
    generator_errors: dict[str, str],
    scorer: str,
) -> Path:
    """
    Recompute analysis CSVs and replace figures/ from generated and scored data.

    This is the single Python function that both normal pipeline completion and
    cached figure rebuilds should call.
    """
    from chorus.pipeline import (
        _build_palette,
        _save_chemical_cluster_umap,
        _save_ecfp_family_landscape,
        _save_ecfp_family_tree,
        _save_score_summary_plot,
        _write_ecfp_family_outputs,
        _write_pocket_distribution_metrics,
        _write_pocket_tanimoto_analysis,
        _write_presentation_analysis_csvs,
        _write_scaffold_family_summary,
        _write_score_correlation_metrics,
        _write_source_pocket_predictability,
        _write_standard_analysis_csvs,
        generate_all_figures,
    )

    scored_df = _write_pocket_tanimoto_analysis(run_dir, generated_df, scored_df, cfg)
    ranked = scored_df.copy()
    ranked["_sort"] = pd.to_numeric(ranked.get("rank_score", 0.0), errors="coerce").fillna(0.0)
    top_hits = (
        ranked.sort_values(
            ["_sort", "n_generators", "n_pockets", "qed"],
            ascending=[False, False, False, False],
        )
        .head(16)
        .copy()
    )
    top_hits.to_csv(run_dir / "top_unique_hits.csv", index=False)

    _write_standard_analysis_csvs(run_dir, generated_df)
    _write_pocket_distribution_metrics(run_dir, generated_df, cfg)
    _write_score_correlation_metrics(run_dir, scored_df)
    _write_source_pocket_predictability(run_dir, generated_df, cfg)
    clustered = _write_presentation_analysis_csvs(run_dir, scored_df, cfg)
    ecfp_families = _write_ecfp_family_outputs(run_dir, scored_df, cfg)
    _write_scaffold_family_summary(run_dir, scored_df, cfg)

    figures_dir = run_dir / "figures"
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    palette = _build_palette(sorted(set(generated_df["generator"])))
    generate_all_figures(
        generated_df=generated_df,
        unique_df=scored_df,
        top_hits=top_hits,
        pocket_specs=pocket_specs,
        palette=palette,
        cfg=cfg,
        out_dir=figures_dir,
    )
    _save_chemical_cluster_umap(clustered, cfg, figures_dir / "chemical_cluster_umap.png")
    _save_ecfp_family_landscape(ecfp_families, cfg, figures_dir / "ecfp_family_landscape.png")
    _save_ecfp_family_tree(run_dir, figures_dir / "ecfp_family_tree.png")
    _save_score_summary_plot(
        run_dir / "cluster_summary.csv",
        figures_dir / "morgan_basin_score_summary.png",
        id_col="cluster_id",
        title="Morgan fingerprint basin score summary",
        label_prefix="C",
    )
    _save_score_summary_plot(
        run_dir / "ecfp_group_summary.csv",
        figures_dir / "ecfp_group_score_summary.png",
        id_col="ecfp_group",
        title="Hierarchical ECFP group score summary",
        label_prefix="G",
    )
    _save_score_summary_plot(
        run_dir / "scaffold_family_summary.csv",
        figures_dir / "scaffold_family_score_summary.png",
        id_col="scaffold_family",
        title="Scaffold family score summary",
        label_prefix="F",
    )
    _save_chemical_space_overview(figures_dir)

    summary = {
        "run_dir": str(run_dir),
        "run_name": run_name,
        "pdb_id": cfg.pdb_id,
        "target_name": cfg.target_name,
        "generators": list(cfg.generators),
        "generator_errors": generator_errors,
        "n_pockets": len(pocket_specs),
        "pocket_ids": [s.pocket_id for s in pocket_specs],
        "pocket_sources": [s.pocket_source for s in pocket_specs],
        "n_generated_rows": int(len(generated_df)),
        "n_unique_molecules": (
            int(scored_df["smiles"].nunique()) if "smiles" in scored_df.columns else int(len(scored_df))
        ),
        "scorer": scorer,
        "top_smiles": top_hits.iloc[0]["smiles"] if not top_hits.empty else None,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return run_dir


def _save_chemical_space_overview(figures_dir: Path) -> None:
    """Combine the main chemical-space views into one presentation-ready image."""
    panels = [
        ("Aggregated PCA: generator, best pocket, score", figures_dir / "ligand_space_pca.png"),
        ("Source-pocket PCA: generator, conditioning pocket, score", figures_dir / "source_pocket_ligand_space_pca.png"),
        ("Aggregated UMAP: generator, best pocket, score", figures_dir / "ligand_space_umap.png"),
        ("Source-pocket UMAP: generator, conditioning pocket, score", figures_dir / "source_pocket_ligand_space_umap.png"),
        ("Pocket Tanimoto landscape", figures_dir / "pocket_tanimoto_landscape.png"),
        ("Scaffold family space", figures_dir / "scaffold_family_space.png"),
        ("Hierarchical ECFP families", figures_dir / "ecfp_family_landscape.png"),
    ]
    existing = []
    for title, path in panels:
        if path.exists():
            image = mpimg.imread(path)
            existing.append((title, image))
    if not existing:
        return

    width_inches = 18.0
    title_inches = 0.45
    row_heights = []
    for _, image in existing:
        height_px, width_px = image.shape[:2]
        row_heights.append(max(2.6, width_inches * height_px / max(width_px, 1) + title_inches))

    fig, axes = plt.subplots(
        len(existing),
        1,
        figsize=(width_inches, sum(row_heights) + 0.8),
        gridspec_kw={"height_ratios": row_heights},
    )
    if len(existing) == 1:
        axes = [axes]

    for ax, (title, image) in zip(axes, existing):
        ax.imshow(image)
        ax.set_title(title, fontsize=15, pad=8, loc="left")
        ax.axis("off")

    fig.suptitle("Chemical-space analysis overview", fontsize=18)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(figures_dir / "chemical_space_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
