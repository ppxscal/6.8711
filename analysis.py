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
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", Path(__file__).resolve().parent / "cache" / "matplotlib"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
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
        _save_ecfp_family_landscape,
        _save_ecfp_family_tree,
        _save_score_summary_plot,
        _write_rdkit_image,
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

    scored_df = _add_ra_scores(run_dir, scored_df)
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
    scaffold_families = _write_scaffold_family_summary(run_dir, scored_df, cfg)

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
    _save_aggregate_pca_2x2(scored_df, palette, cfg, figures_dir / "aggregate_pca_2x2.png")
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
    _save_score_distribution_summary(scored_df, figures_dir / "score_distribution_summary.png")
    representative_rows = _write_representative_cluster_smiles(
        run_dir=run_dir,
        clustered=clustered,
        ecfp_families=ecfp_families,
        scaffold_families=scaffold_families,
        seed=cfg.seed,
    )
    _save_representative_cluster_grid(
        representative_rows,
        figures_dir / "representative_cluster_molecules.png",
        _write_rdkit_image,
    )
    _save_representative_cluster_grid(
        representative_rows,
        figures_dir / "representative_ecfp_groups.png",
        _write_rdkit_image,
        label_types={"ECFP group"},
    )
    _save_representative_cluster_grid(
        representative_rows,
        figures_dir / "representative_scaffold_families.png",
        _write_rdkit_image,
        label_types={"Scaffold family"},
        smiles_col="top_scaffold",
        fallback_smiles_col="medoid_smiles",
    )
    _save_embedding_overviews(figures_dir)
    _save_family_structure_overview(figures_dir)
    _save_diagnostic_overview(figures_dir)
    _save_main_story_overview(figures_dir)
    _save_chemical_space_overview(figures_dir)
    _organize_figure_sections(figures_dir)

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


def _add_ra_scores(run_dir: Path, scored_df: pd.DataFrame) -> pd.DataFrame:
    """Add cached Reymond-group RAscore values when the optional package is installed."""
    out = scored_df.copy()
    if out.empty or "smiles" not in out.columns:
        return out

    cache_path = run_dir / "ra_scores.csv"
    if cache_path.exists():
        try:
            cache = pd.read_csv(cache_path)
            if {"smiles", "ra_score"}.issubset(cache.columns):
                if pd.to_numeric(cache["ra_score"], errors="coerce").notna().any():
                    out = out.merge(cache[["smiles", "ra_score"]].drop_duplicates("smiles"), on="smiles", how="left")
                    return out
                print("RA score cache exists but contains no scores; recomputing.", flush=True)
        except Exception as exc:
            print(f"WARNING: could not read RA score cache: {exc}", flush=True)

    print(f"Computing RA scores for {out['smiles'].nunique()} unique molecules ...", flush=True)
    smiles = out["smiles"].dropna().astype(str).drop_duplicates().tolist()

    cache = _score_ra_with_external_python(smiles)
    if cache is not None:
        cache.to_csv(cache_path, index=False)
        return out.merge(cache, on="smiles", how="left")

    scorer = _build_ra_scorer()
    if scorer is None:
        out["ra_score"] = np.nan
        pd.DataFrame({"smiles": smiles, "ra_score": np.nan}).to_csv(cache_path, index=False)
        return out

    rows = []
    for smi in smiles:
        try:
            score = float(scorer.predict(smi))
        except Exception:
            score = np.nan
        rows.append({"smiles": smi, "ra_score": score})
    cache = pd.DataFrame(rows)
    cache.to_csv(cache_path, index=False)
    out = out.merge(cache, on="smiles", how="left")
    return out


def _score_ra_with_external_python(smiles: list[str]) -> pd.DataFrame | None:
    candidates = []
    if os.environ.get("RASCORE_PYTHON"):
        candidates.append(Path(os.environ["RASCORE_PYTHON"]))
    candidates.append(Path(__file__).resolve().parent / "envs" / "uv" / "rascore" / "bin" / "python")

    script = r'''
import pandas as pd
import sys
from RAscore import RAscore_XGB

inp, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(inp)
scorer = RAscore_XGB.RAScorerXGB()
rows = []
for smi in df["smiles"].dropna().astype(str):
    try:
        score = float(scorer.predict(smi))
    except Exception:
        score = float("nan")
    rows.append({"smiles": smi, "ra_score": score})
pd.DataFrame(rows).to_csv(out, index=False)
'''
    for python in candidates:
        if not python.exists():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            inp = tmpdir / "smiles.csv"
            out = tmpdir / "ra_scores.csv"
            pd.DataFrame({"smiles": smiles}).to_csv(inp, index=False)
            result = subprocess.run(
                [str(python), "-c", script, str(inp), str(out)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode == 0 and out.exists():
                return pd.read_csv(out)
            print(
                f"WARNING: external RAscore failed with {python}: "
                f"{result.stderr.strip()[:500]}",
                flush=True,
            )
    return None


def _build_ra_scorer():
    try:
        from RAscore import RAscore_XGB  # type: ignore[import-not-found]
    except Exception:
        print(
            "WARNING: Reymond-group RAscore is not installed; "
            "RA score panel will be marked unavailable.",
            flush=True,
        )
        return None
    try:
        return RAscore_XGB.RAScorerXGB()
    except Exception as exc:
        print(f"WARNING: could not initialize RAscore XGB model: {exc}", flush=True)
        return None


def _save_chemical_space_overview(figures_dir: Path) -> None:
    """Compact chemical-space overview for the single aggregate PCA reduction."""
    panels = [
        ("Aggregate PCA overview", figures_dir / "aggregate_pca_overview.png"),
    ]
    _stack_existing_images(
        panels,
        figures_dir / "chemical_space_overview.png",
        title="Aggregate chemical-space overview (PCA)",
        width_inches=18.0,
    )


def _save_embedding_overviews(figures_dir: Path) -> None:
    _stack_existing_images(
        [("Aggregate unique-molecule PCA", figures_dir / "aggregate_pca_2x2.png")],
        figures_dir / "aggregate_pca_overview.png",
        title="Aggregate PCA: generator, best pocket, RTMScore, RA score",
        width_inches=18.0,
    )


def _save_family_structure_overview(figures_dir: Path) -> None:
    panels = [
        ("Scaffold family landscape", figures_dir / "scaffold_family_space.png"),
        ("Hierarchical ECFP family landscape", figures_dir / "ecfp_family_landscape.png"),
        ("Representative scaffold-family cores", figures_dir / "representative_scaffold_families.png"),
        ("Representative ECFP-group molecules", figures_dir / "representative_ecfp_groups.png"),
    ]
    _stack_existing_images(
        panels,
        figures_dir / "family_structure_overview.png",
        title="Scaffold and ECFP family interpretation",
        width_inches=17.0,
    )


def _save_diagnostic_overview(figures_dir: Path) -> None:
    panels = [
        ("Source-pocket PCA", figures_dir / "source_pocket_ligand_space_pca.png"),
        ("Pocket Tanimoto neighborhood diagnostics", figures_dir / "pocket_tanimoto_landscape.png"),
    ]
    _stack_existing_images(
        panels,
        figures_dir / "diagnostic_overview.png",
        title="Pocket-conditioning diagnostics",
        width_inches=17.0,
    )


def _save_main_story_overview(figures_dir: Path) -> None:
    """Compact figure stack for the report/presentation main text."""
    panels = [
        ("1. RTMScore distributions by generator and best-scoring pocket",
         figures_dir / "score_distribution_summary.png"),
        ("2. Chemical space: generator bias, best pocket, and RTMScore",
         figures_dir / "aggregate_pca_2x2.png"),
        ("3. Hierarchical ECFP modules: chemical families and RTMScore",
         figures_dir / "ecfp_family_landscape.png"),
        ("4. Representative high-scoring chemical groups",
         figures_dir / "representative_cluster_molecules.png"),
    ]
    _stack_existing_images(
        panels,
        figures_dir / "main_story_overview.png",
        title="Per-target generation and scoring summary",
        width_inches=17.0,
    )


def _save_aggregate_pca_2x2(
    scored_df: pd.DataFrame,
    palette: dict[str, str],
    cfg: "Config",
    out_path: Path,
) -> None:
    """Aggregate unique-molecule PCA colored by generator, pocket, RTMScore, and RA score."""
    if scored_df.empty or "smiles" not in scored_df.columns:
        return
    from chorus.pipeline import PCA, fp_array

    if PCA is None:
        return

    plot_df = scored_df.copy()
    if len(plot_df) > cfg.max_pca_points:
        plot_df = plot_df.sample(cfg.max_pca_points, random_state=cfg.seed)

    fps, rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(str(row["smiles"]))
        if arr is not None:
            fps.append(arr)
            rows.append(row)
    if len(fps) < 3:
        return

    embed_df = pd.DataFrame(rows).reset_index(drop=True)
    coords = PCA(n_components=2, random_state=cfg.seed).fit_transform(np.stack(fps))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]

    fig = plt.figure(figsize=(16.5, 12))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 0.045, 1.0, 0.045],
        height_ratios=[1.0, 1.0],
        wspace=0.18,
        hspace=0.26,
    )
    axes = np.array([
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 2]),
    ])
    cax_score = fig.add_subplot(gs[1, 1])
    cax_ra = fig.add_subplot(gs[1, 3])

    ax = axes[0]
    for gen in sorted(embed_df["generators"].dropna().astype(str).unique()):
        sub = embed_df[embed_df["generators"].astype(str) == gen]
        ax.scatter(sub["x"], sub["y"], s=14, alpha=0.58, c=palette.get(gen, "#4E79A7"), label=gen)
    ax.set_title("Generator")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    pocket_col = "rtmscore_best_pocket_id" if "rtmscore_best_pocket_id" in embed_df.columns else "pocket_ids"
    pockets = sorted(embed_df[pocket_col].dropna().astype(str).unique()) if pocket_col in embed_df else []
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(pockets), 1)))
    for i, pocket in enumerate(pockets):
        sub = embed_df[embed_df[pocket_col].astype(str) == pocket]
        ax.scatter(sub["x"], sub["y"], s=14, alpha=0.62, color=colors[i], label=pocket)
    ax.set_title("Best-scoring pocket")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    score = pd.to_numeric(embed_df.get("rank_score", np.nan), errors="coerce")
    if score.notna().any():
        sc = ax.scatter(embed_df["x"], embed_df["y"], s=14, alpha=0.7, c=score, cmap="RdYlGn")
        fig.colorbar(sc, cax=cax_score, label="RTMScore")
    else:
        ax.scatter(embed_df["x"], embed_df["y"], s=14, alpha=0.35, color="#999999")
        cax_score.axis("off")
    ax.set_title("RTMScore")

    ax = axes[3]
    ra = pd.to_numeric(embed_df.get("ra_score", np.nan), errors="coerce")
    if ra.notna().any():
        sc = ax.scatter(embed_df["x"], embed_df["y"], s=14, alpha=0.7, c=ra, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(sc, cax=cax_ra, label="RA score")
    else:
        ax.scatter(embed_df["x"], embed_df["y"], s=14, alpha=0.35, color="#999999")
        ax.text(0.5, 0.5, "RA score unavailable", transform=ax.transAxes, ha="center", va="center")
        cax_ra.axis("off")
    ax.set_title("Retrosynthetic accessibility")

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    fig.suptitle("Aggregate ligand chemical space (PCA on Morgan fingerprints)", fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _stack_existing_images(
    panels: list[tuple[str, Path]],
    out_path: Path,
    *,
    title: str,
    width_inches: float = 18.0,
) -> None:
    existing = []
    for panel_title, path in panels:
        if path.exists():
            existing.append((panel_title, mpimg.imread(path)))
    if not existing:
        return

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

    for ax, (panel_title, image) in zip(axes, existing):
        ax.imshow(image)
        ax.set_title(panel_title, fontsize=15, pad=8, loc="left")
        ax.axis("off")

    fig.suptitle(title, fontsize=18)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_score_distribution_summary(scored_df: pd.DataFrame, out_path: Path) -> None:
    if scored_df.empty or "rank_score" not in scored_df.columns:
        return

    df = scored_df.copy()
    df["rank_score"] = pd.to_numeric(df["rank_score"], errors="coerce")
    df = df[df["rank_score"].notna()]
    if df.empty:
        return

    generator_col = "rtmscore_best_generator" if "rtmscore_best_generator" in df.columns else "generators"
    pocket_col = "rtmscore_best_pocket_id" if "rtmscore_best_pocket_id" in df.columns else "pocket_ids"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    _boxplot_by_category(
        axes[0],
        df,
        generator_col,
        "rank_score",
        "Score by best-pose generator",
        "Generator",
    )
    _boxplot_by_category(
        axes[1],
        df,
        pocket_col,
        "rank_score",
        "Score by best-pose pocket",
        "Pocket",
    )

    ax = axes[2]
    top_n = max(1, int(np.ceil(0.05 * len(df))))
    top = df.sort_values("rank_score", ascending=False).head(top_n)
    labels = []
    values = []
    for label, sub in top.groupby(generator_col, sort=True):
        labels.append(str(label))
        values.append(len(sub) / top_n)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 1)))
    ax.bar(labels, values, color=colors[:len(labels)])
    ax.set_ylim(0, max(values) * 1.2 if values else 1.0)
    ax.set_ylabel("Fraction of top 5%")
    ax.set_title("Top-score enrichment")
    ax.tick_params(axis="x", rotation=35)

    fig.suptitle("RTMScore distribution summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _boxplot_by_category(
    ax,
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    xlabel: str,
) -> None:
    if category_col not in df.columns:
        ax.axis("off")
        return
    working = df[[category_col, value_col]].dropna().copy()
    working[category_col] = working[category_col].astype(str)
    order = (
        working.groupby(category_col)[value_col]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    values = [working.loc[working[category_col] == label, value_col].to_numpy(float) for label in order]
    ax.boxplot(values, labels=[f"{label}\n(n={len(vals)})" for label, vals in zip(order, values)], showfliers=False)
    means = [float(np.mean(vals)) if len(vals) else np.nan for vals in values]
    ax.scatter(range(1, len(means) + 1), means, s=28, color="#E15759", zorder=3, label="mean")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RTMScore")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, fontsize=8)


def _write_representative_cluster_smiles(
    run_dir: Path,
    clustered: pd.DataFrame,
    ecfp_families: pd.DataFrame,
    scaffold_families: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rows.extend(_representative_rows(clustered, "Morgan basin", "chemical_cluster", "C", seed))
    rows.extend(_representative_rows(ecfp_families, "ECFP group", "ecfp_group", "G", seed))
    rows.extend(_representative_rows(scaffold_families, "Scaffold family", "scaffold_family", "F", seed))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["max_rank_score", "mean_rank_score", "n_molecules"],
            ascending=[False, False, False],
            na_position="last",
        )
    out.to_csv(run_dir / "representative_cluster_smiles.csv", index=False)
    return out


def _representative_rows(
    df: pd.DataFrame,
    label_type: str,
    label_col: str,
    prefix: str,
    seed: int,
    max_labels: int = 12,
    examples_per_label: int = 6,
) -> list[dict[str, object]]:
    if df.empty or label_col not in df.columns or "smiles" not in df.columns:
        return []

    work = df.copy()
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce").fillna(-1).astype(int)
    work = work[work[label_col] != -1].copy()
    if work.empty:
        return []
    work["rank_score"] = pd.to_numeric(work.get("rank_score", np.nan), errors="coerce")

    summary = (
        work.groupby(label_col)
        .agg(
            n_molecules=("smiles", "nunique"),
            mean_rank_score=("rank_score", "mean"),
            median_rank_score=("rank_score", "median"),
            max_rank_score=("rank_score", "max"),
        )
        .reset_index()
        .sort_values(["max_rank_score", "mean_rank_score", "n_molecules"], ascending=[False, False, False])
        .head(max_labels)
    )

    rows = []
    for label_id in summary[label_col].astype(int):
        sub = work[work[label_col] == label_id].copy()
        sub = sub.sort_values("rank_score", ascending=False, na_position="last")
        scaffolds = sub["scaffold"].dropna().astype(str) if "scaffold" in sub.columns else pd.Series(dtype=str)
        top_scaffold = scaffolds.mode().iloc[0] if not scaffolds.empty else ""
        top_examples = sub["smiles"].dropna().astype(str).drop_duplicates().head(examples_per_label).tolist()
        medoid = _medoid_smiles(sub["smiles"].dropna().astype(str).drop_duplicates().tolist(), seed=seed)
        best = top_examples[0] if top_examples else ""
        rows.append({
            "label_type": label_type,
            "label_id": f"{prefix}{label_id}",
            "raw_label_id": int(label_id),
            "n_molecules": int(sub["smiles"].nunique()),
            "mean_rank_score": float(pd.to_numeric(sub["rank_score"], errors="coerce").mean()),
            "median_rank_score": float(pd.to_numeric(sub["rank_score"], errors="coerce").median()),
            "max_rank_score": float(pd.to_numeric(sub["rank_score"], errors="coerce").max()),
            "best_score_smiles": best,
            "medoid_smiles": medoid,
            "top_scaffold": top_scaffold,
            "generator_composition": _composition_string_local(sub.get("generators", pd.Series(dtype=str))),
            "pocket_composition": _composition_string_local(sub.get("pocket_ids", pd.Series(dtype=str))),
            "sampled_top_smiles": " ; ".join(top_examples),
        })
    return rows


def _medoid_smiles(smiles: list[str], seed: int, max_mols: int = 200) -> str:
    if not smiles:
        return ""
    if len(smiles) == 1:
        return smiles[0]
    from chorus.pipeline import _compute_tanimoto_matrix, morgan_fp

    sample = pd.Series(smiles).drop_duplicates()
    if len(sample) > max_mols:
        sample = sample.sample(max_mols, random_state=seed)
    sample_smiles = sample.tolist()
    fps = [morgan_fp(smi) for smi in sample_smiles]
    keep = [(smi, fp) for smi, fp in zip(sample_smiles, fps) if fp is not None]
    if not keep:
        return sample_smiles[0]
    kept_smiles, kept_fps = zip(*keep)
    sim = _compute_tanimoto_matrix(list(kept_fps), list(kept_fps))
    return str(kept_smiles[int(np.argmax(sim.mean(axis=1)))])


def _composition_string_local(values: pd.Series, limit: int = 4) -> str:
    from collections import Counter

    counts: Counter[str] = Counter()
    for value in values.dropna():
        for part in str(value).split(","):
            part = part.strip()
            if part:
                counts[part] += 1
    return "; ".join(f"{key}:{count}" for key, count in counts.most_common(limit))


def _save_representative_cluster_grid(
    representative_rows: pd.DataFrame,
    out_path: Path,
    write_image,
    max_mols: int = 12,
    label_types: set[str] | None = None,
    smiles_col: str = "best_score_smiles",
    fallback_smiles_col: str = "medoid_smiles",
) -> None:
    if representative_rows.empty:
        return
    from rdkit.Chem import Draw
    from chorus.pipeline import mol_from_smiles

    rows = representative_rows.copy()
    if label_types is not None and "label_type" in rows.columns:
        rows = rows[rows["label_type"].isin(label_types)]
    rows = rows.head(max_mols)
    if rows.empty:
        return
    mols = []
    legends = []
    for row in rows.itertuples(index=False):
        smiles = getattr(row, smiles_col, "") or getattr(row, fallback_smiles_col, "")
        mol = mol_from_smiles(str(smiles))
        if mol is None:
            fallback = getattr(row, fallback_smiles_col, "")
            mol = mol_from_smiles(str(fallback))
        if mol is None:
            continue
        mols.append(mol)
        legends.append(
            f"{row.label_type} {row.label_id}\n"
            f"max={row.max_rank_score:.1f}, n={int(row.n_molecules)}"
        )
    if not mols:
        return
    image = Draw.MolsToGridImage(
        mols,
        legends=legends,
        molsPerRow=4,
        subImgSize=(330, 250),
        useSVG=False,
    )
    write_image(image, out_path)


def _organize_figure_sections(figures_dir: Path) -> None:
    """Move detailed figures into topic folders and leave a compact root index."""
    sections: dict[str, list[str]] = {
        "00_overview": [
            "main_story_overview.png",
            "chemical_space_overview.png",
            "aggregate_pca_overview.png",
        ],
        "01_scores": [
            "score_distribution_summary.png",
            "ranked_top_hits.png",
            "top_hits_grid.png",
            "pocket_druggability.png",
            "pocket_generator_heatmap.png",
            "yield_matrix.png",
            "morgan_basin_score_summary.png",
            "ecfp_group_score_summary.png",
            "scaffold_family_score_summary.png",
        ],
        "02_chemical_space": [
            "aggregate_pca_2x2.png",
        ],
        "03_families": [
            "family_structure_overview.png",
            "ecfp_family_landscape.png",
            "ecfp_family_tree.png",
            "scaffold_family_space.png",
            "scaffold_family_pocket_heatmap.png",
            "representative_cluster_molecules.png",
            "representative_ecfp_groups.png",
            "representative_scaffold_families.png",
        ],
        "04_diagnostics": [
            "diagnostic_overview.png",
            "source_pocket_ligand_space_pca.png",
            "pocket_tanimoto_landscape.png",
            "summary_dashboard.png",
        ],
    }
    for section, filenames in sections.items():
        section_dir = figures_dir / section
        section_dir.mkdir(exist_ok=True)
        for filename in filenames:
            src = figures_dir / filename
            if src.exists():
                dst = section_dir / filename
                if dst.exists():
                    dst.unlink()
                shutil.move(str(src), str(dst))

    _write_figures_index(figures_dir, sections)


def _write_figures_index(figures_dir: Path, sections: dict[str, list[str]]) -> None:
    descriptions = {
        "00_overview": "Start here: compact presentation and aggregate PCA overview.",
        "01_scores": "Score distributions, top hits, pocket/generator yield, and score summaries.",
        "02_chemical_space": "Aggregate unique-molecule PCA colored by generator, best pocket, RTMScore, and RA score.",
        "03_families": "Scaffold/ECFP family landscapes and representative molecule or scaffold renderings.",
        "04_diagnostics": "Source-pocket and Tanimoto diagnostics for pocket-conditioning behavior.",
    }
    lines = [
        "# Figure Index",
        "",
        "This directory is organized by analysis question. The detailed PNGs are grouped into section folders.",
        "",
    ]
    for section in sections:
        section_dir = figures_dir / section
        existing = [path.name for path in sorted(section_dir.glob("*.png"))] if section_dir.exists() else []
        if not existing:
            continue
        lines.append(f"## {section}")
        lines.append("")
        lines.append(descriptions.get(section, ""))
        lines.append("")
        for filename in existing:
            lines.append(f"- `{section}/{filename}`")
        lines.append("")
    (figures_dir / "README.md").write_text("\n".join(lines))
