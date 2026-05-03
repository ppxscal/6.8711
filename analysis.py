"""Analysis rebuilds for completed Chorus runs.

This module owns the public path for turning generated/scored CSV caches into
summary tables and figures. Fresh experiments and cached rebuilds call the same
functions so analysis output remains reproducible.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, TYPE_CHECKING

MATPLOTLIB_CONFIG_DIR = Path(
    os.environ.get("MPLCONFIGDIR", Path(__file__).resolve().parent / "cache" / "matplotlib")
)
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import Draw

from chorus.generators import fp_array, get_scaffold, mol_from_smiles, morgan_fp

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from chorus.config import Config
    from chorus.pockets import PocketSpec


def scored_candidates_path(run_dir: Path, scorer: str) -> Path:
    scorer = scorer.strip().lower()
    return run_dir / f"scored_candidates_{scorer}.csv"


def load_cached_run_tables(
    run_dir: Path,
    scorer: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list["PocketSpec"]] | None:
    """Load cached generated/unique/scored tables if a run is analysis-ready."""
    from chorus.pockets import read_cached_pocket_specs

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
    if scorer == "rtmscore":
        has_scores = (
            "rtmscore_score" in scored_df.columns
            and pd.to_numeric(scored_df["rtmscore_score"], errors="coerce").notna().any()
        )
        has_pose_counts = (
            "rtmscore_n_poses" in scored_df.columns
            and pd.to_numeric(scored_df["rtmscore_n_poses"], errors="coerce").fillna(0).sum() > 0
        )
        if not (has_scores and has_pose_counts):
            print(
                f"Cached {scored_path.name} has no usable RTMScore scores; rerunning scoring.",
                flush=True,
            )
            return None
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

    scored_df = add_primary_source_labels(generated_df, scored_df)
    scored_df = add_ra_scores(run_dir, scored_df)
    scored_df = write_pocket_tanimoto_analysis(run_dir, generated_df, scored_df, cfg)
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

    write_standard_analysis_csvs(run_dir, generated_df)
    write_pocket_distribution_metrics(run_dir, generated_df, cfg)
    write_score_correlation_metrics(run_dir, scored_df)
    write_source_pocket_predictability(run_dir, generated_df, cfg)
    clustered = write_presentation_analysis_csvs(run_dir, scored_df, cfg)
    ecfp_families = write_ecfp_family_outputs(run_dir, scored_df, cfg)
    scaffold_families = write_scaffold_family_summary(run_dir, scored_df, cfg)

    figures_dir = run_dir / "figures"
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    palette = build_generator_palette(sorted(set(generated_df["generator"])))
    generate_all_figures(
        generated_df=generated_df,
        unique_df=scored_df,
        top_hits=top_hits,
        pocket_specs=pocket_specs,
        palette=palette,
        cfg=cfg,
        out_dir=figures_dir,
    )
    save_aggregate_pca_2x2(scored_df, palette, cfg, figures_dir / "aggregate_pca_2x2.png")
    save_ecfp_family_landscape(ecfp_families, cfg, figures_dir / "ecfp_family_landscape.png")
    save_ecfp_family_tree(run_dir, figures_dir / "ecfp_family_tree.png")
    save_score_summary_plot(
        run_dir / "cluster_summary.csv",
        figures_dir / "morgan_basin_score_summary.png",
        id_col="cluster_id",
        title="Morgan fingerprint basin score summary",
        label_prefix="C",
    )
    save_score_summary_plot(
        run_dir / "ecfp_group_summary.csv",
        figures_dir / "ecfp_group_score_summary.png",
        id_col="ecfp_group",
        title="Hierarchical ECFP group score summary",
        label_prefix="G",
    )
    save_score_summary_plot(
        run_dir / "scaffold_family_summary.csv",
        figures_dir / "scaffold_family_score_summary.png",
        id_col="scaffold_family",
        title="Scaffold family score summary",
        label_prefix="F",
    )
    save_score_distribution_summary(scored_df, figures_dir / "score_distribution_summary.png")
    representative_rows = write_representative_cluster_smiles(
        run_dir=run_dir,
        clustered=clustered,
        ecfp_families=ecfp_families,
        scaffold_families=scaffold_families,
        seed=cfg.seed,
    )
    save_representative_cluster_grid(
        representative_rows,
        figures_dir / "representative_cluster_molecules.png",
        write_rdkit_image,
    )
    save_representative_cluster_grid(
        representative_rows,
        figures_dir / "representative_ecfp_groups.png",
        write_rdkit_image,
        label_types={"ECFP group"},
    )
    save_representative_cluster_grid(
        representative_rows,
        figures_dir / "representative_scaffold_families.png",
        write_rdkit_image,
        label_types={"Scaffold family"},
        smiles_col="top_scaffold",
        fallback_smiles_col="medoid_smiles",
    )
    save_embedding_overviews(figures_dir)
    save_family_structure_overview(figures_dir)
    save_diagnostic_overview(figures_dir)
    save_main_story_overview(figures_dir)
    save_chemical_space_overview(figures_dir)
    organize_figure_sections(figures_dir)

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


def add_ra_scores(run_dir: Path, scored_df: pd.DataFrame) -> pd.DataFrame:
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

    cache = score_ra_with_external_python(smiles)
    if cache is not None:
        cache.to_csv(cache_path, index=False)
        return out.merge(cache, on="smiles", how="left")

    scorer = build_ra_scorer()
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


def score_ra_with_external_python(smiles: list[str]) -> pd.DataFrame | None:
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


def build_ra_scorer():
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


def save_chemical_space_overview(figures_dir: Path) -> None:
    """Compact chemical-space overview for the single aggregate PCA reduction."""
    panels = [
        ("Aggregate PCA overview", figures_dir / "aggregate_pca_overview.png"),
    ]
    stack_existing_images(
        panels,
        figures_dir / "chemical_space_overview.png",
        title="Aggregate chemical-space overview (PCA)",
        width_inches=18.0,
    )


def save_embedding_overviews(figures_dir: Path) -> None:
    stack_existing_images(
        [("Aggregate unique-molecule PCA", figures_dir / "aggregate_pca_2x2.png")],
        figures_dir / "aggregate_pca_overview.png",
        title="Aggregate PCA: generator, conditioning pocket, RTMScore, RA score",
        width_inches=18.0,
    )


def save_family_structure_overview(figures_dir: Path) -> None:
    panels = [
        ("Scaffold family landscape", figures_dir / "scaffold_family_space.png"),
        ("Hierarchical ECFP family landscape", figures_dir / "ecfp_family_landscape.png"),
        ("Representative scaffold-family cores", figures_dir / "representative_scaffold_families.png"),
        ("Representative ECFP-group molecules", figures_dir / "representative_ecfp_groups.png"),
    ]
    stack_existing_images(
        panels,
        figures_dir / "family_structure_overview.png",
        title="Scaffold and ECFP family interpretation",
        width_inches=17.0,
    )


def save_diagnostic_overview(figures_dir: Path) -> None:
    panels = [
        ("Source-pocket PCA", figures_dir / "source_pocket_ligand_space_pca.png"),
        ("Pocket Tanimoto neighborhood diagnostics", figures_dir / "pocket_tanimoto_landscape.png"),
    ]
    stack_existing_images(
        panels,
        figures_dir / "diagnostic_overview.png",
        title="Pocket-conditioning diagnostics",
        width_inches=17.0,
    )


def save_main_story_overview(figures_dir: Path) -> None:
    """Compact figure stack for the report/presentation main text."""
    panels = [
        ("1. RTMScore distributions by generator and conditioning pocket",
         figures_dir / "score_distribution_summary.png"),
        ("2. Chemical space: generator bias, conditioning pocket, and RTMScore",
         figures_dir / "aggregate_pca_2x2.png"),
        ("3. Hierarchical ECFP modules: chemical families and RTMScore",
         figures_dir / "ecfp_family_landscape.png"),
        ("4. Representative high-scoring chemical groups",
         figures_dir / "representative_cluster_molecules.png"),
    ]
    stack_existing_images(
        panels,
        figures_dir / "main_story_overview.png",
        title="Per-target generation and scoring summary",
        width_inches=17.0,
    )


def save_aggregate_pca_2x2(
    scored_df: pd.DataFrame,
    palette: dict[str, str],
    cfg: "Config",
    out_path: Path,
) -> None:
    """Aggregate unique-molecule PCA colored by generator, pocket, RTMScore, and RA score."""
    if scored_df.empty or "smiles" not in scored_df.columns:
        return
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
    embed_df["generator_label"] = embed_df.get(
        "primary_generator", embed_df.get("generators", pd.Series(index=embed_df.index))
    ).apply(
        generator_provenance_label
    )
    for label in sorted(embed_df["generator_label"].dropna().astype(str).unique()):
        sub = embed_df[embed_df["generator_label"].astype(str) == label]
        ax.scatter(sub["x"], sub["y"], s=14, alpha=0.58, c=palette.get(label, "#4E79A7"), label=label)
    ax.set_title("Generator")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    embed_df["conditioning_pocket_label"] = embed_df.get(
        "primary_pocket_id", embed_df.get("pocket_ids", pd.Series(index=embed_df.index))
    ).apply(
        conditioning_pocket_label
    )
    pocket_col = "conditioning_pocket_label"
    pockets = sorted(embed_df[pocket_col].dropna().astype(str).unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(pockets), 1)))
    for i, pocket in enumerate(pockets):
        sub = embed_df[embed_df[pocket_col].astype(str) == pocket]
        ax.scatter(sub["x"], sub["y"], s=14, alpha=0.62, color=colors[i], label=pocket)
    ax.set_title("Conditioning pocket")
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


def stack_existing_images(
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


def save_score_distribution_summary(scored_df: pd.DataFrame, out_path: Path) -> None:
    if scored_df.empty or "rank_score" not in scored_df.columns:
        return

    df = scored_df.copy()
    df["rank_score"] = pd.to_numeric(df["rank_score"], errors="coerce")
    df = df[df["rank_score"].notna()]
    if df.empty:
        return

    if "rtmscore_best_generator" in df.columns:
        generator_col = "best_generator_label"
        df[generator_col] = df["rtmscore_best_generator"].apply(display_generator_name)
    else:
        generator_col = "generator_label"
        df[generator_col] = df.get(
            "primary_generator", df.get("generators", pd.Series(index=df.index))
        ).apply(generator_provenance_label)
    pocket_col = "conditioning_pocket_label"
    df[pocket_col] = df.get(
        "primary_pocket_id", df.get("pocket_ids", pd.Series(index=df.index))
    ).apply(conditioning_pocket_label)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    boxplot_by_category(
        axes[0],
        df,
        generator_col,
        "rank_score",
        "Score by best-pose generator",
        "Generator",
    )
    boxplot_by_category(
        axes[1],
        df,
        pocket_col,
        "rank_score",
        "Score by conditioning pocket",
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


def boxplot_by_category(
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


def write_representative_cluster_smiles(
    run_dir: Path,
    clustered: pd.DataFrame,
    ecfp_families: pd.DataFrame,
    scaffold_families: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rows.extend(representative_rows(clustered, "Morgan basin", "chemical_cluster", "C", seed))
    rows.extend(representative_rows(ecfp_families, "ECFP group", "ecfp_group", "G", seed))
    rows.extend(representative_rows(scaffold_families, "Scaffold family", "scaffold_family", "F", seed))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["max_rank_score", "mean_rank_score", "n_molecules"],
            ascending=[False, False, False],
            na_position="last",
        )
    out.to_csv(run_dir / "representative_cluster_smiles.csv", index=False)
    return out


def representative_rows(
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
        medoid = medoid_smiles(sub["smiles"].dropna().astype(str).drop_duplicates().tolist(), seed=seed)
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
            "generator_composition": composition_string(sub.get("generators", pd.Series(dtype=str))),
            "pocket_composition": composition_string(sub.get("pocket_ids", pd.Series(dtype=str))),
            "sampled_top_smiles": " ; ".join(top_examples),
        })
    return rows


def medoid_smiles(smiles: list[str], seed: int, max_mols: int = 200) -> str:
    if not smiles:
        return ""
    if len(smiles) == 1:
        return smiles[0]
    sample = pd.Series(smiles).drop_duplicates()
    if len(sample) > max_mols:
        sample = sample.sample(max_mols, random_state=seed)
    sample_smiles = sample.tolist()
    fps = [morgan_fp(smi) for smi in sample_smiles]
    keep = [(smi, fp) for smi, fp in zip(sample_smiles, fps) if fp is not None]
    if not keep:
        return sample_smiles[0]
    kept_smiles, kept_fps = zip(*keep)
    sim = compute_tanimoto_matrix(list(kept_fps), list(kept_fps))
    return str(kept_smiles[int(np.argmax(sim.mean(axis=1)))])


def save_representative_cluster_grid(
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


def organize_figure_sections(figures_dir: Path) -> None:
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

    write_figures_index(figures_dir, sections)


def write_figures_index(figures_dir: Path, sections: dict[str, list[str]]) -> None:
    descriptions = {
        "00_overview": "Start here: compact presentation and aggregate PCA overview.",
        "01_scores": "Score distributions, top hits, pocket/generator yield, and score summaries.",
        "02_chemical_space": "Aggregate unique-molecule PCA colored by generator, conditioning pocket, RTMScore, and RA score.",
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


# Analysis tables
# ---------------------------------------------------------------------------

def build_scaffold_diversity_matrix(generated_df: pd.DataFrame) -> pd.DataFrame:
    generators = sorted(generated_df["generator"].unique())
    scaffold_sets = {
        g: set(generated_df.loc[(generated_df["generator"] == g) & (generated_df["scaffold"] != ""), "scaffold"])
        for g in generators
    }
    mat = {}
    for g1 in generators:
        mat[g1] = {}
        for g2 in generators:
            union = scaffold_sets[g1] | scaffold_sets[g2]
            inter = scaffold_sets[g1] & scaffold_sets[g2]
            mat[g1][g2] = len(inter) / len(union) if union else 0.0
    return pd.DataFrame(mat, index=generators)


def pocket_jaccard_matrix(pocket_df: pd.DataFrame) -> pd.DataFrame:
    generators = sorted(pocket_df["generator"].unique())
    scaffold_sets = {
        g: set(pocket_df.loc[(pocket_df["generator"] == g) & (pocket_df["scaffold"] != ""), "scaffold"])
        for g in generators
    }
    mat = {}
    for g1 in generators:
        mat[g1] = {}
        for g2 in generators:
            union = scaffold_sets[g1] | scaffold_sets[g2]
            inter = scaffold_sets[g1] & scaffold_sets[g2]
            mat[g1][g2] = len(inter) / len(union) if union else 0.0
    return pd.DataFrame(mat, index=generators)


def pocket_sensitivity(generated_df: pd.DataFrame) -> pd.DataFrame:
    return generated_df.groupby(["generator", "pocket_id"])["smiles"].nunique().rename("n_unique")


def compute_budget_summary(generated_df: pd.DataFrame) -> pd.DataFrame:
    return generated_df.groupby("generator").agg(
        total_mols=("smiles", "count"),
        unique_mols=("smiles", "nunique"),
        unique_scaffolds=("scaffold", lambda s: (s != "").sum()),
        mean_qed=("qed", "mean"),
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def ensure_plot_dependencies() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed.")


def generate_all_figures(
    generated_df: pd.DataFrame,
    unique_df: pd.DataFrame,
    top_hits: pd.DataFrame,
    pocket_specs: list[PocketSpec],
    palette: dict[str, str],
    cfg: Config,
    out_dir: Path,
    scored_df: pd.DataFrame | None = None,
) -> None:
    """Write the baseline figure set used by both fresh runs and rebuilds."""
    ensure_plot_dependencies()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_pocket_dir = out_dir / "per_pocket"
    per_pocket_dir.mkdir(exist_ok=True)

    save_top_hits_grid(top_hits, out_dir / "top_hits_grid.png")
    save_ranked_hits(top_hits, out_dir / "ranked_top_hits.png")
    save_source_pocket_ligand_space(
        generated_df, unique_df, palette, cfg, out_dir / "source_pocket_ligand_space_pca.png"
    )
    save_pocket_tanimoto_landscape(unique_df, cfg, out_dir / "pocket_tanimoto_landscape.png")
    save_summary_dashboard(generated_df, unique_df, top_hits, pocket_specs, palette, cfg,
                            out_dir / "summary_dashboard.png")
    save_pocket_generator_heatmap(generated_df, out_dir / "pocket_generator_heatmap.png")
    save_yield_matrix(generated_df, out_dir / "yield_matrix.png")
    save_pocket_druggability(unique_df, pocket_specs, out_dir / "pocket_druggability.png")

    # Scaffold-family plots are interpretive, so keep failures non-fatal.
    if not unique_df.empty and "scaffold" in unique_df.columns:
        try:
            clustered = cluster_scaffolds(
                unique_df,
                max_points=cfg.max_cluster_points,
                seed=cfg.seed,
            )
            save_scaffold_family_space(
                clustered, cfg, out_dir / "scaffold_family_space.png"
            )
            save_scaffold_family_pocket_heatmap(
                clustered, pocket_specs, out_dir / "scaffold_family_pocket_heatmap.png"
            )
        except Exception as exc:
            print(f"WARNING: Scaffold family analysis failed: {exc}")

    ranking_col = (
        "rank_score"
        if "rank_score" in unique_df.columns and unique_df["rank_score"].notna().any()
        else None
    )
    for spec in pocket_specs:
        pocket_mols = unique_df[unique_df["pocket_ids"].str.contains(spec.pocket_id, na=False)]
        if not pocket_mols.empty and ranking_col:
            pocket_top = pocket_mols.sort_values(ranking_col, ascending=False).head(4)
            save_top_hits_grid(pocket_top, per_pocket_dir / f"{spec.pocket_id}_top_hits.png")


def write_rdkit_image(image, out_path: Path) -> None:
    if hasattr(image, "save"):
        image.save(str(out_path))
    elif isinstance(image, (bytes, bytearray)):
        out_path.write_bytes(bytes(image))
    elif hasattr(image, "data"):
        out_path.write_bytes(image.data)


def save_top_hits_grid(top_hits: pd.DataFrame, out_path: Path) -> None:
    ensure_plot_dependencies()
    if top_hits.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No top hits available", ha="center", va="center")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    mols, legends = [], []
    for _, row in top_hits.iterrows():
        mol = mol_from_smiles(row["smiles"])
        if mol is None:
            continue
        mols.append(mol)
        gen = row.get("generators", "")
        legends.append(f"{gen}\nScore={row.get('rank_score', 0):.2f}")
    if mols:
        image = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=4,
                                     subImgSize=(320, 240), useSVG=False)
        write_rdkit_image(image, out_path)


def save_ranked_hits(top_hits: pd.DataFrame, out_path: Path) -> None:
    ensure_plot_dependencies()
    fig, ax = plt.subplots(figsize=(10, 5))
    if not top_hits.empty and "rank_score" in top_hits.columns:
        labels = [f"Hit {i + 1}" for i in range(len(top_hits))]
        scores = top_hits["rank_score"].fillna(0).astype(float)
        ax.bar(labels, scores, color="#4E79A7")
        ax.set_ylabel("Ranking score")
        ax.set_title("Top ranked molecules")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No scored hits available", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_ligand_space(
    unique_df: pd.DataFrame, palette: dict[str, str], cfg: Config, out_path: Path,
    generated_df: pd.DataFrame | None = None,
) -> None:
    """Aggregated unique-molecule PCA colored by generator, conditioning pocket, and score."""
    ensure_plot_dependencies()
    if PCA is None:
        return

    if unique_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    # Compute PCA once on a sample of unique molecules
    plot_df = unique_df.copy()
    if len(plot_df) > cfg.max_pca_points:
        plot_df = plot_df.sample(cfg.max_pca_points, random_state=cfg.seed)

    fps, keep_rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            keep_rows.append(row)
    if not fps:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    embed_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    coords = PCA(n_components=2, random_state=cfg.seed).fit_transform(np.stack(fps))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]
    embed_df["conditioning_pocket_label"] = embed_df.get(
        "primary_pocket_id", embed_df.get("pocket_ids", pd.Series(index=embed_df.index))
    ).apply(conditioning_pocket_label)

    has_score = "rank_score" in embed_df.columns and embed_df["rank_score"].notna().any()
    has_conditioning_pocket = "conditioning_pocket_label" in embed_df.columns
    has_recurrence = "n_pockets" in embed_df.columns

    n_panels = 1 + int(has_conditioning_pocket or has_recurrence) + int(has_score)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: colored by generator
    ax = axes[0]
    embed_df["generator_label"] = embed_df.get(
        "primary_generator", embed_df.get("generators", pd.Series(index=embed_df.index))
    ).apply(
        generator_provenance_label
    )
    for label in sorted(embed_df["generator_label"].dropna().unique()):
        sub = embed_df[embed_df["generator_label"] == label]
        ax.scatter(sub["x"], sub["y"], s=18, alpha=0.6, c=palette.get(label, "#4E79A7"), label=label)
    ax.set_title("Colored by generator")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=8)

    panel = 1

    if has_conditioning_pocket:
        ax = axes[panel]; panel += 1
        pocket_ids = sorted(embed_df["conditioning_pocket_label"].dropna().unique())
        pocket_colors = plt.cm.tab10(np.linspace(0, 1, max(len(pocket_ids), 1)))
        pocket_palette = {pid: pocket_colors[i] for i, pid in enumerate(pocket_ids)}
        for pid in pocket_ids:
            sub = embed_df[embed_df["conditioning_pocket_label"] == pid]
            ax.scatter(sub["x"], sub["y"], s=18, alpha=0.6,
                       color=pocket_palette[pid], label=pid)
        ax.set_title("Conditioning pocket")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(frameon=False, fontsize=8)
    elif has_recurrence:
        ax = axes[panel]; panel += 1
        recurrence = pd.to_numeric(embed_df["n_pockets"], errors="coerce").fillna(1)
        sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.7,
                        c=recurrence, cmap="viridis")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Source pockets rediscovering SMILES")
        ax.set_title("Rediscovery across source pockets")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # Panel 3: colored by ranking score (continuous colormap)
    if has_score:
        ax = axes[panel]
        scores = embed_df["rank_score"].fillna(0).astype(float)
        sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.7,
                        c=scores, cmap="RdYlGn")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Ranking score")
        ax.set_title("Colored by ranking score")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    fig.suptitle("Ligand chemical space (PCA on Morgan fingerprints)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_ligand_space_umap(
    unique_df: pd.DataFrame, palette: dict[str, str], cfg: Config, out_path: Path,
) -> None:
    """Three-panel ligand space UMAP on Morgan fingerprints, if umap-learn is installed."""
    ensure_plot_dependencies()
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError:
        print("WARNING: umap-learn is not installed; skipping ligand_space_umap.png", flush=True)
        return

    if unique_df.empty:
        return

    plot_df = unique_df.copy()
    if len(plot_df) > cfg.max_umap_points:
        plot_df = plot_df.sample(cfg.max_umap_points, random_state=cfg.seed)

    fps, keep_rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            keep_rows.append(row)
    if len(fps) < 3:
        return

    embed_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    print(f"Building ligand-space UMAP on {len(fps)} sampled molecules ...", flush=True)
    reducer = umap.UMAP(
        n_components=2,
        metric="jaccard",
        random_state=cfg.seed,
        n_neighbors=30,
        min_dist=0.1,
    )
    coords = reducer.fit_transform(np.stack(fps).astype(bool))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]
    embed_df["conditioning_pocket_label"] = embed_df.get(
        "primary_pocket_id", embed_df.get("pocket_ids", pd.Series(index=embed_df.index))
    ).apply(conditioning_pocket_label)

    has_score = "rank_score" in embed_df.columns and embed_df["rank_score"].notna().any()
    has_conditioning_pocket = "conditioning_pocket_label" in embed_df.columns
    has_recurrence = "n_pockets" in embed_df.columns
    n_panels = 1 + int(has_conditioning_pocket or has_recurrence) + int(has_score)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    embed_df["generator_label"] = embed_df.get(
        "primary_generator", embed_df.get("generators", pd.Series(index=embed_df.index))
    ).apply(
        generator_provenance_label
    )
    for label in sorted(embed_df["generator_label"].dropna().unique()):
        sub = embed_df[embed_df["generator_label"] == label]
        ax.scatter(sub["x"], sub["y"], s=18, alpha=0.6, c=palette.get(label, "#4E79A7"), label=label)
    ax.set_title("Colored by generator")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.legend(frameon=False, fontsize=8)

    panel = 1
    if has_conditioning_pocket:
        ax = axes[panel]; panel += 1
        pocket_ids = sorted(embed_df["conditioning_pocket_label"].dropna().unique())
        pocket_colors = plt.cm.tab10(np.linspace(0, 1, max(len(pocket_ids), 1)))
        pocket_palette = {pid: pocket_colors[i] for i, pid in enumerate(pocket_ids)}
        for pid in pocket_ids:
            sub = embed_df[embed_df["conditioning_pocket_label"] == pid]
            ax.scatter(sub["x"], sub["y"], s=18, alpha=0.6, color=pocket_palette[pid], label=pid)
        ax.set_title("Conditioning pocket")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.legend(frameon=False, fontsize=8)
    elif has_recurrence:
        ax = axes[panel]; panel += 1
        recurrence = pd.to_numeric(embed_df["n_pockets"], errors="coerce").fillna(1)
        sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.7, c=recurrence, cmap="viridis")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Source pockets rediscovering SMILES")
        ax.set_title("Rediscovery across source pockets")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    if has_score:
        ax = axes[panel]
        scores = embed_df["rank_score"].fillna(0).astype(float)
        sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.7, c=scores, cmap="RdYlGn")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Ranking score")
        ax.set_title("Colored by ranking score")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    fig.suptitle("Ligand chemical space (UMAP on Morgan fingerprints)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def scored_source_dataframe(generated_df: pd.DataFrame, unique_df: pd.DataFrame) -> pd.DataFrame:
    if generated_df.empty or unique_df.empty or "smiles" not in generated_df.columns:
        return generated_df.copy()
    score_cols = [
        col for col in [
            "smiles", "rank_score", "scorer", "rtmscore_score", "rtmscore_mean_score",
            "rtmscore_best_pocket_id", "n_pockets", "n_generators",
            "pocket_tanimoto_entropy", "pocket_tanimoto_z_entropy",
            "pocket_tanimoto_z_specificity", "pocket_tanimoto_z_dominant_pocket",
        ]
        if col in unique_df.columns
    ]
    if len(score_cols) <= 1:
        return generated_df.copy()
    return generated_df.merge(unique_df[score_cols].drop_duplicates("smiles"), on="smiles", how="left")


def save_source_pocket_ligand_space(
    generated_df: pd.DataFrame,
    unique_df: pd.DataFrame,
    palette: dict[str, str],
    cfg: Config,
    out_path: Path,
) -> None:
    """Source-conditioned PCA: one row per generator-pocket molecule, colored by true source pocket."""
    ensure_plot_dependencies()
    if PCA is None or generated_df.empty:
        return

    source_df = scored_source_dataframe(generated_df, unique_df)
    plot_df = source_df.copy()
    if len(plot_df) > cfg.max_pca_points:
        plot_df = plot_df.sample(cfg.max_pca_points, random_state=cfg.seed)

    fps, keep_rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            keep_rows.append(row)
    if len(fps) < 3:
        return

    embed_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    coords = PCA(n_components=2, random_state=cfg.seed).fit_transform(np.stack(fps))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]
    has_score = "rank_score" in embed_df.columns and embed_df["rank_score"].notna().any()

    fig, axes = plt.subplots(1, 3 if has_score else 2, figsize=(24 if has_score else 16, 7))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax = axes[0]
    for gen in sorted(embed_df["generator"].dropna().unique()):
        sub = embed_df[embed_df["generator"] == gen]
        label = display_generator_name(gen)
        ax.scatter(sub["x"], sub["y"], s=18, alpha=0.55, c=palette.get(label, "#4E79A7"), label=label)
    ax.set_title("Source rows colored by generator")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    pocket_ids = sorted(embed_df["pocket_id"].dropna().unique())
    pocket_colors = plt.cm.tab10(np.linspace(0, 1, max(len(pocket_ids), 1)))
    pocket_palette = {pid: pocket_colors[i] for i, pid in enumerate(pocket_ids)}
    for pid in pocket_ids:
        sub = embed_df[embed_df["pocket_id"] == pid]
        ax.scatter(sub["x"], sub["y"], s=18, alpha=0.62, color=pocket_palette[pid], label=pid)
    ax.set_title("True conditioning pocket")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=8)

    if has_score:
        ax = axes[2]
        scores = pd.to_numeric(embed_df["rank_score"], errors="coerce")
        sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.68, c=scores, cmap="RdYlGn")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Ranking score")
        ax.set_title("Unique-molecule score projected onto source rows")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    fig.suptitle("Source-conditioned ligand chemical space (PCA on Morgan fingerprints)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_source_pocket_ligand_space_umap(
    generated_df: pd.DataFrame,
    unique_df: pd.DataFrame,
    palette: dict[str, str],
    cfg: Config,
    out_path: Path,
) -> None:
    ensure_plot_dependencies()
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError:
        return
    if generated_df.empty:
        return

    source_df = scored_source_dataframe(generated_df, unique_df)
    plot_df = source_df.copy()
    if len(plot_df) > cfg.max_umap_points:
        plot_df = plot_df.sample(cfg.max_umap_points, random_state=cfg.seed)

    fps, keep_rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            keep_rows.append(row)
    if len(fps) < 3:
        return

    embed_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    print(f"Building source-pocket UMAP on {len(fps)} sampled source rows ...", flush=True)
    coords = umap.UMAP(
        n_components=2,
        metric="jaccard",
        random_state=cfg.seed,
        n_neighbors=30,
        min_dist=0.1,
    ).fit_transform(np.stack(fps).astype(bool))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]
    has_score = "rank_score" in embed_df.columns and embed_df["rank_score"].notna().any()

    fig, axes = plt.subplots(1, 3 if has_score else 2, figsize=(24 if has_score else 16, 7))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax = axes[0]
    for gen in sorted(embed_df["generator"].dropna().unique()):
        sub = embed_df[embed_df["generator"] == gen]
        label = display_generator_name(gen)
        ax.scatter(sub["x"], sub["y"], s=18, alpha=0.58, c=palette.get(label, "#4E79A7"), label=label)
    ax.set_title("Source rows colored by generator")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    pocket_ids = sorted(embed_df["pocket_id"].dropna().unique())
    pocket_colors = plt.cm.tab10(np.linspace(0, 1, max(len(pocket_ids), 1)))
    pocket_palette = {pid: pocket_colors[i] for i, pid in enumerate(pocket_ids)}
    for pid in pocket_ids:
        sub = embed_df[embed_df["pocket_id"] == pid]
        ax.scatter(sub["x"], sub["y"], s=18, alpha=0.65, color=pocket_palette[pid], label=pid)
    ax.set_title("True conditioning pocket")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.legend(frameon=False, fontsize=8)

    if has_score:
        ax = axes[2]
        scores = pd.to_numeric(embed_df["rank_score"], errors="coerce")
        sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.7, c=scores, cmap="RdYlGn")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Ranking score")
        ax.set_title("Unique-molecule score projected onto source rows")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    fig.suptitle("Source-conditioned ligand chemical space (UMAP on Morgan fingerprints)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_pocket_tanimoto_landscape(
    unique_df: pd.DataFrame, cfg: Config, out_path: Path,
) -> None:
    """PCA landscape colored by pocket-neighborhood Tanimoto specificity."""
    ensure_plot_dependencies()
    needed = {"smiles", "pocket_tanimoto_entropy", "pocket_tanimoto_dominant_pocket"}
    if PCA is None or unique_df.empty or not needed.issubset(unique_df.columns):
        return

    plot_df = unique_df.copy()
    if len(plot_df) > cfg.max_pca_points:
        plot_df = plot_df.sample(cfg.max_pca_points, random_state=cfg.seed)

    fps, keep_rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            keep_rows.append(row)
    if len(fps) < 3:
        return

    embed_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    coords = PCA(n_components=2, random_state=cfg.seed).fit_transform(np.stack(fps))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    ax = axes[0]
    dominant_col = (
        "pocket_tanimoto_z_dominant_pocket"
        if "pocket_tanimoto_z_dominant_pocket" in embed_df.columns
        else "pocket_tanimoto_dominant_pocket"
    )
    entropy_col = (
        "pocket_tanimoto_z_entropy"
        if "pocket_tanimoto_z_entropy" in embed_df.columns
        else "pocket_tanimoto_entropy"
    )
    margin_col = (
        "pocket_tanimoto_z_margin"
        if "pocket_tanimoto_z_margin" in embed_df.columns
        else "pocket_tanimoto_margin"
    )

    pockets = sorted(p for p in embed_df[dominant_col].dropna().unique() if str(p))
    pocket_colors = plt.cm.tab10(np.linspace(0, 1, max(len(pockets), 1)))
    pocket_palette = {pid: pocket_colors[i] for i, pid in enumerate(pockets)}
    for pid in pockets:
        sub = embed_df[embed_df[dominant_col] == pid]
        ax.scatter(sub["x"], sub["y"], s=18, alpha=0.62, color=pocket_palette[pid], label=pid)
    ax.set_title("Nearest pocket chemical neighborhood")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    entropy = pd.to_numeric(embed_df[entropy_col], errors="coerce")
    sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.72, c=entropy, cmap="viridis_r", vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Normalized Tanimoto entropy")
    ax.set_title("Low normalized entropy = pocket-specific")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    ax = axes[2]
    margin = pd.to_numeric(embed_df.get(margin_col, 0.0), errors="coerce").fillna(0.0)
    sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.72, c=margin, cmap="magma")
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Best minus second-best pocket score")
    ax.set_title("Pocket-neighborhood specificity margin")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    fig.suptitle("Pocket-specific chemical neighborhoods (Morgan fingerprint PCA)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_chemical_cluster_umap(
    clustered: pd.DataFrame, cfg: Config, out_path: Path,
) -> None:
    """UMAP view of Morgan fingerprint chemical-space basins from HDBSCAN."""
    ensure_plot_dependencies()
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError:
        return
    if clustered.empty or "chemical_cluster" not in clustered.columns:
        return

    plot_df = clustered.copy()
    if len(plot_df) > cfg.max_umap_points:
        plot_df = plot_df.sample(cfg.max_umap_points, random_state=cfg.seed)

    fps, keep_rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            keep_rows.append(row)
    if len(fps) < 3:
        return

    embed_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    print(f"Building chemical-cluster UMAP on {len(fps)} sampled molecules ...", flush=True)
    coords = umap.UMAP(
        n_components=2,
        metric="jaccard",
        random_state=cfg.seed,
        n_neighbors=30,
        min_dist=0.1,
    ).fit_transform(np.stack(fps).astype(bool))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]

    clusters = sorted(int(c) for c in embed_df["chemical_cluster"].dropna().unique())
    real_clusters = [c for c in clusters if c != -1]
    has_score = "rank_score" in embed_df.columns and embed_df["rank_score"].notna().any()
    fig, axes = plt.subplots(1, 1 + int(has_score), figsize=(9 * (1 + int(has_score)), 8))
    if not has_score:
        axes = [axes]

    ax = axes[0]
    noise = embed_df[embed_df["chemical_cluster"] == -1]
    if not noise.empty:
        ax.scatter(noise["x"], noise["y"], s=10, alpha=0.18, color="lightgrey", label=f"unclustered (n={len(noise)})")
    cmap = plt.cm.tab20 if len(real_clusters) <= 20 else plt.cm.hsv
    for i, cluster_id in enumerate(real_clusters):
        sub = embed_df[embed_df["chemical_cluster"] == cluster_id]
        ax.scatter(
            sub["x"], sub["y"], s=22, alpha=0.75,
            color=cmap(i / max(len(real_clusters), 1)),
            label=f"C{cluster_id} (n={len(sub)})",
        )
    ax.set_title(f"Morgan fingerprint basins (HDBSCAN, {len(real_clusters)} clusters)")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    if len(real_clusters) <= 15:
        ax.legend(frameon=False, fontsize=7, ncol=2)

    if has_score:
        ax2 = axes[1]
        scores = embed_df["rank_score"].fillna(0).astype(float)
        sc = ax2.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.7, c=scores, cmap="RdYlGn")
        plt.colorbar(sc, ax=ax2, shrink=0.8, label="Ranking score")
        ax2.set_title("Basins colored by ranking score")
        ax2.set_xlabel("UMAP1"); ax2.set_ylabel("UMAP2")

    fig.suptitle("Chemical-space basins from Morgan fingerprints", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scaffold family clustering + figures
# ---------------------------------------------------------------------------

def cluster_scaffolds(
    unique_df: pd.DataFrame,
    min_cluster_size: int = 8,
    max_points: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Cluster molecules into scaffold families using HDBSCAN on scaffold Morgan FPs.

    Molecules sharing a Bemis-Murcko scaffold cluster are structurally related —
    this gives 'scaffold families' even when exact scaffolds differ.
    Returns unique_df with a new 'scaffold_family' column (int, -1 = noise/singleton).
    """
    import hdbscan

    out = unique_df.copy()
    out["scaffold_family"] = -1

    scaffold_col = out["scaffold"].fillna("")
    # Build per-scaffold fingerprints (cluster scaffolds, propagate to molecules)
    scaffold_counts = scaffold_col[scaffold_col != ""].value_counts()
    unique_scaffolds = scaffold_counts.index.tolist()
    if max_points is not None and len(unique_scaffolds) > max_points:
        unique_scaffolds = unique_scaffolds[:max_points]
        print(
            f"Clustering scaffold families on {len(unique_scaffolds)}/"
            f"{len(scaffold_counts)} most frequent scaffolds ...",
            flush=True,
        )
    else:
        print(f"Clustering scaffold families on {len(unique_scaffolds)} scaffolds ...", flush=True)
    if len(unique_scaffolds) < 2:
        return out

    fps = []
    valid_scaffolds = []
    for sc in unique_scaffolds:
        fp = fp_array(sc)
        if fp is not None:
            fps.append(fp)
            valid_scaffolds.append(sc)

    if len(fps) < 2:
        return out

    X = np.stack(fps)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="jaccard",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)
    scaffold_to_family = dict(zip(valid_scaffolds, labels.tolist()))
    out["scaffold_family"] = scaffold_col.map(scaffold_to_family).fillna(-1).astype(int)
    return out


def cluster_molecule_fingerprints(
    unique_df: pd.DataFrame,
    min_cluster_size: int = 25,
    max_points: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign broad Morgan-fingerprint basins for presentation summaries."""
    """Cluster molecules into Morgan fingerprint chemical-space basins."""
    import hdbscan

    out = unique_df.copy()
    out["chemical_cluster"] = -1
    if out.empty:
        return out

    cluster_df = out
    if max_points is not None and len(cluster_df) > max_points:
        cluster_df = cluster_df.sample(max_points, random_state=seed)
        print(
            f"Clustering Morgan fingerprints for presentation CSVs "
            f"on {len(cluster_df)}/{len(out)} sampled molecules ...",
            flush=True,
        )
    else:
        print(
            f"Clustering Morgan fingerprints for presentation CSVs "
            f"on {len(cluster_df)} molecules ...",
            flush=True,
        )

    fps, indices = [], []
    for idx, row in cluster_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            indices.append(idx)
    if len(fps) < max(2, min_cluster_size):
        return out

    X = np.stack(fps).astype(float)
    if PCA is not None and len(fps) > 3:
        n_components = min(50, X.shape[1], len(fps) - 1)
        X_cluster = PCA(n_components=n_components, random_state=seed).fit_transform(X)
        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric="euclidean",
            cluster_selection_method="eom",
        ).fit_predict(X_cluster)
    else:
        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="jaccard",
            cluster_selection_method="eom",
        ).fit_predict(X.astype(bool))
    out.loc[indices, "chemical_cluster"] = labels.astype(int)
    return out


def split_multi_value(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def first_multi_value(value: Any) -> str:
    values = split_multi_value(value)
    return values[0] if values else "unknown"


GENERATOR_DISPLAY_NAMES = {
    "DiffSBDD": "DiffSBDD cond",
    "DiffSBDDJoint": "DiffSBDD joint",
}


def display_generator_name(name: Any) -> str:
    text = str(name)
    return GENERATOR_DISPLAY_NAMES.get(text, text)


def generator_provenance_label(value: Any) -> str:
    return display_generator_name(first_multi_value(value))


def conditioning_pocket_label(value: Any) -> str:
    return first_multi_value(value)


def add_primary_source_labels(generated_df: pd.DataFrame, unique_df: pd.DataFrame) -> pd.DataFrame:
    """Attach one source generator and pocket per unique molecule for aggregate plots."""
    if unique_df.empty or generated_df.empty or "smiles" not in unique_df.columns:
        return unique_df.copy()

    required = {"smiles", "generator", "pocket_id"}
    if not required.issubset(generated_df.columns):
        return unique_df.copy()

    sort_cols = ["smiles"]
    if "source_rank" in generated_df.columns:
        sort_cols.append("source_rank")
    sort_cols.extend(["generator", "pocket_id"])

    primary = (
        generated_df.sort_values(sort_cols)
        .drop_duplicates("smiles")
        [["smiles", "generator", "pocket_id"]]
        .rename(columns={"generator": "primary_generator", "pocket_id": "primary_pocket_id"})
    )
    out = unique_df.drop(
        columns=[c for c in ("primary_generator", "primary_pocket_id") if c in unique_df.columns]
    ).merge(primary, on="smiles", how="left")
    return out


def composition_string(values: pd.Series, limit: int = 4) -> str:
    from collections import Counter

    counts: Counter[str] = Counter()
    for value in values.dropna():
        for part in split_multi_value(value):
            counts[part] += 1
    return "; ".join(f"{key}:{count}" for key, count in counts.most_common(limit))


def write_cluster_summary(clustered: pd.DataFrame, out_path: Path) -> None:
    if clustered.empty or "chemical_cluster" not in clustered.columns:
        pd.DataFrame().to_csv(out_path, index=False)
        return

    rows = []
    score_col = "rank_score" if "rank_score" in clustered.columns else None
    for cluster_id, sub in clustered.groupby("chemical_cluster", sort=True):
        if int(cluster_id) == -1:
            continue
        scaffold = ""
        if "scaffold" in sub.columns and sub["scaffold"].notna().any():
            modes = sub["scaffold"].dropna().astype(str)
            scaffold = modes.mode().iloc[0] if not modes.empty else ""
        row = {
            "cluster_id": int(cluster_id),
            "n_molecules": int(len(sub)),
            "top_scaffold": scaffold,
            "generator_composition": composition_string(sub.get("generators", pd.Series(dtype=str))),
            "pocket_composition": composition_string(sub.get("pocket_ids", pd.Series(dtype=str))),
            "mean_qed": float(sub["qed"].mean()) if "qed" in sub.columns else np.nan,
            "mean_mw": float(sub["mw"].mean()) if "mw" in sub.columns else np.nan,
            "mean_logp": float(sub["logp"].mean()) if "logp" in sub.columns else np.nan,
        }
        if score_col:
            scores = pd.to_numeric(sub[score_col], errors="coerce")
            row.update({
                "mean_score": float(scores.mean()),
                "median_score": float(scores.median()),
                "max_score": float(scores.max()),
            })
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame.to_csv(out_path, index=False)
        return
    frame.sort_values(
        ["max_score", "mean_score", "n_molecules"],
        ascending=[False, False, False],
        na_position="last",
    ).to_csv(out_path, index=False)


def write_scaffold_family_summary(run_dir: Path, merged: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    try:
        clustered = cluster_scaffolds(
            merged,
            max_points=cfg.max_cluster_points,
            seed=cfg.seed,
        )
        rows = []
        assigned = clustered[clustered["scaffold_family"] != -1].copy()
        for family_id, sub in assigned.groupby("scaffold_family", sort=True):
            scaffolds = sub["scaffold"].dropna().astype(str) if "scaffold" in sub.columns else pd.Series(dtype=str)
            top_scaffold = scaffolds.mode().iloc[0] if not scaffolds.empty else ""
            scores = pd.to_numeric(sub.get("rank_score", np.nan), errors="coerce")
            rows.append({
                "scaffold_family": int(family_id),
                "n_molecules": int(len(sub)),
                "n_scaffolds": int(scaffolds.nunique()) if not scaffolds.empty else 0,
                "top_scaffold": top_scaffold,
                "generator_composition": composition_string(sub.get("generators", pd.Series(dtype=str))),
                "pocket_composition": composition_string(sub.get("pocket_ids", pd.Series(dtype=str))),
                "mean_score": float(scores.mean()),
                "median_score": float(scores.median()),
                "max_score": float(scores.max()),
                "mean_qed": float(pd.to_numeric(sub.get("qed", np.nan), errors="coerce").mean()),
            })
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values(
                ["max_score", "mean_score", "n_molecules"],
                ascending=[False, False, False],
                na_position="last",
            )
        frame.to_csv(run_dir / "scaffold_family_summary.csv", index=False)
        return clustered
    except Exception as exc:
        print(f"WARNING: scaffold family summary failed: {exc}", flush=True)
        pd.DataFrame().to_csv(run_dir / "scaffold_family_summary.csv", index=False)
        return merged.copy()


def write_top_hit_enrichment(
    df: pd.DataFrame,
    label_col: str,
    out_path: Path,
    label_name: str,
    score_col: str = "rank_score",
) -> None:
    if df.empty or label_col not in df.columns or score_col not in df.columns:
        pd.DataFrame().to_csv(out_path, index=False)
        return

    records = []
    base = df[["smiles", label_col, score_col]].copy()
    base[score_col] = pd.to_numeric(base[score_col], errors="coerce").fillna(0.0)
    exploded_rows = []
    for _, row in base.iterrows():
        if label_col == "chemical_cluster":
            if pd.isna(row[label_col]):
                labels = []
            else:
                labels = [str(int(row[label_col]))]
        else:
            labels = split_multi_value(row[label_col])
        for label in labels:
            if label and label != "-1":
                exploded_rows.append({"smiles": row["smiles"], label_name: label, score_col: row[score_col]})
    exploded = pd.DataFrame(exploded_rows)
    if exploded.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return

    totals = exploded.groupby(label_name)["smiles"].nunique()
    ranked = base.sort_values(score_col, ascending=False)
    for frac in (0.01, 0.05, 0.10):
        n_top = max(1, int(np.ceil(len(ranked) * frac)))
        top_smiles = set(ranked.head(n_top)["smiles"])
        top = exploded[exploded["smiles"].isin(top_smiles)]
        top_counts = top.groupby(label_name)["smiles"].nunique()
        for label, total in totals.items():
            n_label_top = int(top_counts.get(label, 0))
            records.append({
                "top_fraction": frac,
                label_name: label,
                "n_top": n_label_top,
                "n_total": int(total),
                "fraction_of_top_hits": n_label_top / n_top,
                "top_hit_rate_within_label": n_label_top / int(total) if total else 0.0,
            })
    pd.DataFrame(records).sort_values(
        ["top_fraction", "n_top", "top_hit_rate_within_label"],
        ascending=[True, False, False],
    ).to_csv(out_path, index=False)


def mean_topk_tanimoto(query_fp: Any, ref_fps: list[Any], top_k: int) -> float:
    if not ref_fps:
        return 0.0
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps), dtype=float)
    if sims.size == 0:
        return 0.0
    k = max(1, min(int(top_k), sims.size))
    if sims.size > k:
        sims = np.partition(sims, -k)[-k:]
    return float(np.mean(sims))


def write_pocket_tanimoto_analysis(
    run_dir: Path,
    generated_df: pd.DataFrame,
    merged: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Measure whether each molecule is closest to one pocket-specific neighborhood."""
    out = merged.copy()
    if out.empty or "smiles" not in out.columns or "pocket_id" not in generated_df.columns:
        return out

    pocket_refs: dict[str, list[Any]] = {}
    for pocket_id, sub in generated_df.groupby("pocket_id", sort=True):
        smiles = sub["smiles"].dropna().astype(str).drop_duplicates()
        if cfg.max_tanimoto_refs_per_pocket and len(smiles) > cfg.max_tanimoto_refs_per_pocket:
            smiles = smiles.sample(cfg.max_tanimoto_refs_per_pocket, random_state=cfg.seed)
        fps = [fp for fp in (morgan_fp(smi) for smi in smiles) if fp is not None]
        if fps:
            pocket_refs[str(pocket_id)] = fps

    pocket_ids = sorted(pocket_refs)
    if len(pocket_ids) < 2:
        return out

    print(
        "Computing pocket Tanimoto entropy "
        f"({len(out)} molecules, {len(pocket_ids)} pockets, "
        f"up to {cfg.max_tanimoto_refs_per_pocket} refs/pocket) ...",
        flush=True,
    )

    sim_cols = {pid: f"tanimoto_to_{pid}" for pid in pocket_ids}
    for col in sim_cols.values():
        out[col] = np.nan
    out["pocket_tanimoto_entropy"] = np.nan
    out["pocket_tanimoto_specificity"] = np.nan
    out["pocket_tanimoto_dominant_pocket"] = ""
    out["pocket_tanimoto_max"] = np.nan
    out["pocket_tanimoto_second"] = np.nan
    out["pocket_tanimoto_margin"] = np.nan
    out["pocket_tanimoto_z_entropy"] = np.nan
    out["pocket_tanimoto_z_specificity"] = np.nan
    out["pocket_tanimoto_z_dominant_pocket"] = ""
    out["pocket_tanimoto_z_margin"] = np.nan

    denom = np.log(len(pocket_ids))
    eps = 1e-12
    for idx, row in out.iterrows():
        fp = morgan_fp(str(row["smiles"]))
        if fp is None:
            continue
        sims = np.asarray([
            mean_topk_tanimoto(fp, pocket_refs[pid], cfg.tanimoto_top_k)
            for pid in pocket_ids
        ], dtype=float)
        for pid, value in zip(pocket_ids, sims):
            out.at[idx, sim_cols[pid]] = value
        order = np.argsort(sims)[::-1]
        best = float(sims[order[0]])
        second = float(sims[order[1]]) if len(order) > 1 else 0.0
        total = float(sims.sum())
        if total > eps:
            probs = sims / total
            entropy = float(-(probs * np.log(probs + eps)).sum() / denom)
        else:
            entropy = np.nan
        out.at[idx, "pocket_tanimoto_entropy"] = entropy
        out.at[idx, "pocket_tanimoto_specificity"] = 1.0 - entropy if pd.notna(entropy) else np.nan
        out.at[idx, "pocket_tanimoto_dominant_pocket"] = pocket_ids[int(order[0])]
        out.at[idx, "pocket_tanimoto_max"] = best
        out.at[idx, "pocket_tanimoto_second"] = second
        out.at[idx, "pocket_tanimoto_margin"] = best - second

    sim_frame = out[list(sim_cols.values())].apply(pd.to_numeric, errors="coerce")
    means = sim_frame.mean(axis=0)
    stds = sim_frame.std(axis=0).replace(0, np.nan)
    z_frame = (sim_frame - means) / stds
    z_frame = z_frame.fillna(0.0)
    for idx, z_row in z_frame.iterrows():
        z = z_row.to_numpy(dtype=float)
        shifted = z - np.max(z)
        weights = np.exp(shifted)
        probs = weights / max(float(weights.sum()), eps)
        order = np.argsort(z)[::-1]
        entropy = float(-(probs * np.log(probs + eps)).sum() / denom)
        best = float(z[order[0]])
        second = float(z[order[1]]) if len(order) > 1 else 0.0
        out.at[idx, "pocket_tanimoto_z_entropy"] = entropy
        out.at[idx, "pocket_tanimoto_z_specificity"] = 1.0 - entropy
        out.at[idx, "pocket_tanimoto_z_dominant_pocket"] = pocket_ids[int(order[0])]
        out.at[idx, "pocket_tanimoto_z_margin"] = best - second

    analysis_cols = [
        "smiles", "generators", "pocket_ids", "rank_score",
        "pocket_tanimoto_dominant_pocket", "pocket_tanimoto_entropy",
        "pocket_tanimoto_specificity", "pocket_tanimoto_max",
        "pocket_tanimoto_second", "pocket_tanimoto_margin",
        "pocket_tanimoto_z_dominant_pocket", "pocket_tanimoto_z_entropy",
        "pocket_tanimoto_z_specificity", "pocket_tanimoto_z_margin",
        *sim_cols.values(),
    ]
    available_cols = [col for col in analysis_cols if col in out.columns]
    out[available_cols].to_csv(run_dir / "pocket_tanimoto_entropy.csv", index=False)

    source_rows = []
    for _, row in out.iterrows():
        for source_pocket in split_multi_value(row.get("pocket_ids", "")):
            source_rows.append({
                "source_pocket": source_pocket,
                "dominant_tanimoto_pocket": row.get("pocket_tanimoto_dominant_pocket", ""),
                "rank_score": row.get("rank_score", np.nan),
                "entropy": row.get("pocket_tanimoto_entropy", np.nan),
                "specificity": row.get("pocket_tanimoto_specificity", np.nan),
                "margin": row.get("pocket_tanimoto_margin", np.nan),
                "z_dominant_tanimoto_pocket": row.get("pocket_tanimoto_z_dominant_pocket", ""),
                "z_entropy": row.get("pocket_tanimoto_z_entropy", np.nan),
                "z_specificity": row.get("pocket_tanimoto_z_specificity", np.nan),
                "z_margin": row.get("pocket_tanimoto_z_margin", np.nan),
            })
    source_df = pd.DataFrame(source_rows)
    if not source_df.empty:
        summary_rows = []
        for source_pocket, sub in source_df.groupby("source_pocket", sort=True):
            match = sub["dominant_tanimoto_pocket"].astype(str) == str(source_pocket)
            summary_rows.append({
                "source_pocket": source_pocket,
                "n_molecules": int(len(sub)),
                "mean_entropy": float(pd.to_numeric(sub["entropy"], errors="coerce").mean()),
                "mean_specificity": float(pd.to_numeric(sub["specificity"], errors="coerce").mean()),
                "mean_margin": float(pd.to_numeric(sub["margin"], errors="coerce").mean()),
                "mean_z_entropy": float(pd.to_numeric(sub["z_entropy"], errors="coerce").mean()),
                "mean_z_specificity": float(pd.to_numeric(sub["z_specificity"], errors="coerce").mean()),
                "mean_z_margin": float(pd.to_numeric(sub["z_margin"], errors="coerce").mean()),
                "mean_rank_score": float(pd.to_numeric(sub["rank_score"], errors="coerce").mean()),
                "fraction_nearest_to_own_pocket": float(match.mean()),
                "z_fraction_nearest_to_own_pocket": float(
                    (sub["z_dominant_tanimoto_pocket"].astype(str) == str(source_pocket)).mean()
                ),
                "dominant_pocket_composition": composition_string(sub["dominant_tanimoto_pocket"]),
                "z_dominant_pocket_composition": composition_string(sub["z_dominant_tanimoto_pocket"]),
            })
        pd.DataFrame(summary_rows).to_csv(run_dir / "pocket_tanimoto_summary.csv", index=False)
        pd.crosstab(
            source_df["source_pocket"],
            source_df["dominant_tanimoto_pocket"],
        ).to_csv(run_dir / "pocket_tanimoto_confusion.csv")

    return out


def compute_tanimoto_matrix(fps_a: list[Any], fps_b: list[Any]) -> np.ndarray:
    if not fps_a or not fps_b:
        return np.zeros((len(fps_a), len(fps_b)), dtype=float)
    return np.asarray(
        [DataStructs.BulkTanimotoSimilarity(fp, fps_b) for fp in fps_a],
        dtype=float,
    )


def write_pocket_distribution_metrics(run_dir: Path, generated_df: pd.DataFrame, cfg: Config) -> None:
    if generated_df.empty or "pocket_id" not in generated_df.columns:
        pd.DataFrame().to_csv(run_dir / "pocket_chemical_distribution_metrics.csv", index=False)
        return

    pocket_fps: dict[str, list[Any]] = {}
    for pocket_id, sub in generated_df.groupby("pocket_id", sort=True):
        smiles = sub["smiles"].dropna().astype(str).drop_duplicates()
        if cfg.max_tanimoto_refs_per_pocket and len(smiles) > cfg.max_tanimoto_refs_per_pocket:
            smiles = smiles.sample(cfg.max_tanimoto_refs_per_pocket, random_state=cfg.seed)
        fps = [fp for fp in (morgan_fp(smi) for smi in smiles) if fp is not None]
        if fps:
            pocket_fps[str(pocket_id)] = fps

    rows = []
    for source, fps_a in pocket_fps.items():
        for target, fps_b in pocket_fps.items():
            sims = compute_tanimoto_matrix(fps_a, fps_b)
            if sims.size == 0:
                continue
            if source == target and sims.shape[0] == sims.shape[1]:
                np.fill_diagonal(sims, np.nan)
            nearest = np.nanmax(sims, axis=1)
            rows.append({
                "source_pocket": source,
                "target_pocket": target,
                "n_source": int(len(fps_a)),
                "n_target": int(len(fps_b)),
                "mean_nearest_tanimoto": float(np.nanmean(nearest)),
                "median_nearest_tanimoto": float(np.nanmedian(nearest)),
                "p90_nearest_tanimoto": float(np.nanpercentile(nearest, 90)),
                "mean_pairwise_tanimoto": float(np.nanmean(sims)),
            })
    pd.DataFrame(rows).to_csv(run_dir / "pocket_chemical_distribution_metrics.csv", index=False)


def write_score_correlation_metrics(run_dir: Path, merged: pd.DataFrame) -> None:
    if merged.empty or "rank_score" not in merged.columns:
        pd.DataFrame().to_csv(run_dir / "score_correlation_metrics.csv", index=False)
        return
    candidates = [
        "mw", "logp", "qed", "hbd", "hba", "rot_bonds", "rings",
        "n_generators", "n_pockets", "source_min_rank",
        "rtmscore_n_poses", "pocket_tanimoto_entropy",
        "pocket_tanimoto_specificity", "pocket_tanimoto_margin",
        "pocket_tanimoto_z_entropy", "pocket_tanimoto_z_specificity",
        "pocket_tanimoto_z_margin",
    ]
    rows = []
    score = pd.to_numeric(merged["rank_score"], errors="coerce")
    for col in candidates:
        if col not in merged.columns:
            continue
        values = pd.to_numeric(merged[col], errors="coerce")
        valid = score.notna() & values.notna()
        if int(valid.sum()) < 3:
            continue
        rows.append({
            "feature": col,
            "n": int(valid.sum()),
            "pearson_r": float(score[valid].corr(values[valid], method="pearson")),
            "spearman_r": float(score[valid].corr(values[valid], method="spearman")),
        })
    pd.DataFrame(rows).sort_values("spearman_r", key=lambda s: s.abs(), ascending=False).to_csv(
        run_dir / "score_correlation_metrics.csv", index=False
    )


def write_source_pocket_predictability(run_dir: Path, generated_df: pd.DataFrame, cfg: Config) -> None:
    """Estimate whether generated fingerprints retain a source-pocket signal."""
    if generated_df.empty or generated_df["pocket_id"].nunique() < 2:
        pd.DataFrame().to_csv(run_dir / "source_pocket_predictability.csv", index=False)
        return
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except Exception as exc:
        print(f"WARNING: sklearn classifier unavailable; skipping source pocket predictability: {exc}", flush=True)
        pd.DataFrame().to_csv(run_dir / "source_pocket_predictability.csv", index=False)
        return

    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [("all_generators", generated_df)]
    groups.extend((str(gen), sub) for gen, sub in generated_df.groupby("generator", sort=True))
    for label, sub in groups:
        sub = sub.dropna(subset=["smiles", "pocket_id"]).copy()
        if len(sub) > cfg.max_pca_points:
            sub = sub.sample(cfg.max_pca_points, random_state=cfg.seed)
        counts = sub["pocket_id"].value_counts()
        if len(counts) < 2 or counts.min() < 3:
            continue
        fps, labels = [], []
        for row in sub.itertuples(index=False):
            arr = fp_array(str(row.smiles))
            if arr is not None:
                fps.append(arr)
                labels.append(str(row.pocket_id))
        if len(set(labels)) < 2 or len(fps) < 6:
            continue
        X = np.stack(fps)
        y = np.asarray(labels)
        min_class = min(np.unique(y, return_counts=True)[1])
        n_splits = max(2, min(5, int(min_class)))
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
        scores = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy")
        rng = np.random.default_rng(cfg.seed)
        shuffled = y.copy()
        rng.shuffle(shuffled)
        null_scores = cross_val_score(model, X, shuffled, cv=cv, scoring="balanced_accuracy")
        rows.append({
            "subset": label,
            "n": int(len(y)),
            "n_pockets": int(len(set(y))),
            "balanced_accuracy_mean": float(np.mean(scores)),
            "balanced_accuracy_std": float(np.std(scores)),
            "shuffled_balanced_accuracy_mean": float(np.mean(null_scores)),
            "delta_vs_shuffled": float(np.mean(scores) - np.mean(null_scores)),
        })
    pd.DataFrame(rows).to_csv(run_dir / "source_pocket_predictability.csv", index=False)


def butina_ecfp_families(df: pd.DataFrame, cfg: Config, sim_threshold: float | None = None) -> pd.DataFrame:
    from rdkit.ML.Cluster import Butina

    out = df.copy()
    out["ecfp_family"] = -1
    if out.empty:
        return out
    sample = out
    if len(sample) > cfg.max_cluster_points:
        sample = sample.sample(cfg.max_cluster_points, random_state=cfg.seed)
    fps, indices = [], []
    for idx, row in sample.iterrows():
        fp = morgan_fp(str(row["smiles"]))
        if fp is not None:
            fps.append(fp)
            indices.append(idx)
    if len(fps) < 2:
        return out
    distances = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        distances.extend([1.0 - x for x in sims])
    threshold = cfg.ecfp_family_sim_threshold if sim_threshold is None else sim_threshold
    clusters = Butina.ClusterData(distances, len(fps), 1.0 - threshold, isDistData=True)
    for family_id, cluster in enumerate(clusters):
        for local_idx in cluster:
            out.at[indices[int(local_idx)], "ecfp_family"] = int(family_id)
    return out


def write_ecfp_family_outputs(run_dir: Path, merged: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Build ECFP/Tanimoto families and their score summaries."""
    try:
        families = butina_ecfp_families(merged, cfg)
        assigned = families[families["ecfp_family"] != -1].copy()
        rows = []
        for family_id, sub in assigned.groupby("ecfp_family", sort=True):
            scaffold = ""
            if "scaffold" in sub.columns:
                scaffolds = sub["scaffold"].dropna().astype(str)
                scaffold = scaffolds.mode().iloc[0] if not scaffolds.empty else ""
            medoid = sub.sort_values("rank_score", ascending=False)["smiles"].iloc[0] if "rank_score" in sub else sub["smiles"].iloc[0]
            rows.append({
                "ecfp_family": int(family_id),
                "n_molecules": int(len(sub)),
                "representative_smiles": medoid,
                "top_scaffold": scaffold,
                "generator_composition": composition_string(sub.get("generators", pd.Series(dtype=str))),
                "pocket_composition": composition_string(sub.get("pocket_ids", pd.Series(dtype=str))),
                "mean_rank_score": float(pd.to_numeric(sub.get("rank_score", np.nan), errors="coerce").mean()),
                "max_rank_score": float(pd.to_numeric(sub.get("rank_score", np.nan), errors="coerce").max()),
                "mean_qed": float(pd.to_numeric(sub.get("qed", np.nan), errors="coerce").mean()),
            })
        summary = pd.DataFrame(rows).sort_values(
            ["n_molecules", "max_rank_score"], ascending=[False, False], na_position="last"
        )
        families, summary = assign_ecfp_hierarchical_groups(families, summary)
        summary.to_csv(run_dir / "ecfp_family_summary.csv", index=False)
        write_ecfp_group_summary(run_dir, families, summary)
        families.to_csv(run_dir / "ecfp_family_assignments.csv", index=False)
        return families
    except Exception as exc:
        print(f"WARNING: ECFP family analysis failed: {exc}", flush=True)
        pd.DataFrame().to_csv(run_dir / "ecfp_family_summary.csv", index=False)
        return merged.copy()


def assign_ecfp_hierarchical_groups(
    families: pd.DataFrame,
    summary: pd.DataFrame,
    max_groups: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = families.copy()
    summary = summary.copy()
    families["ecfp_group"] = -1
    summary["ecfp_group"] = -1
    if summary.empty or "representative_smiles" not in summary.columns:
        return families, summary

    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except Exception:
        for i, family_id in enumerate(summary["ecfp_family"].astype(int).tolist()):
            group_id = i % max_groups
            summary.loc[summary["ecfp_family"].astype(int) == family_id, "ecfp_group"] = group_id
            families.loc[families["ecfp_family"].astype(int) == family_id, "ecfp_group"] = group_id
        return families, summary

    fps, family_ids = [], []
    for row in summary.itertuples(index=False):
        fp = morgan_fp(str(row.representative_smiles))
        if fp is not None:
            fps.append(fp)
            family_ids.append(int(row.ecfp_family))
    if len(fps) < 2:
        if family_ids:
            families.loc[families["ecfp_family"].isin(family_ids), "ecfp_group"] = 0
            summary.loc[summary["ecfp_family"].isin(family_ids), "ecfp_group"] = 0
        return families, summary

    sim = compute_tanimoto_matrix(fps, fps)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    linked = linkage(squareform(dist, checks=False), method="average")
    n_groups = max(2, min(max_groups, len(family_ids)))
    raw_groups = fcluster(linked, t=n_groups, criterion="maxclust")

    # Stable group ids by the best-scoring/largest family in each group.
    group_members: dict[int, list[int]] = {}
    for family_id, raw_group in zip(family_ids, raw_groups):
        group_members.setdefault(int(raw_group), []).append(family_id)
    group_order = sorted(
        group_members,
        key=lambda gid: (
            -int(summary.loc[summary["ecfp_family"].isin(group_members[gid]), "n_molecules"].sum()),
            -float(summary.loc[summary["ecfp_family"].isin(group_members[gid]), "max_rank_score"].max()),
        ),
    )
    group_map = {raw_group: i for i, raw_group in enumerate(group_order)}
    family_to_group = {
        family_id: group_map[int(raw_group)]
        for family_id, raw_group in zip(family_ids, raw_groups)
    }
    summary["ecfp_group"] = summary["ecfp_family"].astype(int).map(family_to_group).fillna(-1).astype(int)
    families["ecfp_group"] = families["ecfp_family"].astype(int).map(family_to_group).fillna(-1).astype(int)
    return families, summary


def write_ecfp_group_summary(run_dir: Path, families: pd.DataFrame, summary: pd.DataFrame) -> None:
    if families.empty or "ecfp_group" not in families.columns:
        pd.DataFrame().to_csv(run_dir / "ecfp_group_summary.csv", index=False)
        return
    assigned = families[families["ecfp_group"] != -1].copy()
    rows = []
    for group_id, sub in assigned.groupby("ecfp_group", sort=True):
        fam_sub = summary[summary["ecfp_group"] == group_id] if "ecfp_group" in summary else pd.DataFrame()
        rows.append({
            "ecfp_group": int(group_id),
            "n_molecules": int(len(sub)),
            "n_families": int(sub["ecfp_family"].nunique()) if "ecfp_family" in sub else 0,
            "top_families": "; ".join(
                f"F{int(r.ecfp_family)}:n={int(r.n_molecules)}"
                for r in fam_sub.head(6).itertuples(index=False)
            ) if not fam_sub.empty else "",
            "generator_composition": composition_string(sub.get("generators", pd.Series(dtype=str))),
            "pocket_composition": composition_string(sub.get("pocket_ids", pd.Series(dtype=str))),
            "mean_rank_score": float(pd.to_numeric(sub.get("rank_score", np.nan), errors="coerce").mean()),
            "max_rank_score": float(pd.to_numeric(sub.get("rank_score", np.nan), errors="coerce").max()),
        })
    pd.DataFrame(rows).sort_values(
        ["n_molecules", "max_rank_score"], ascending=[False, False], na_position="last"
    ).to_csv(run_dir / "ecfp_group_summary.csv", index=False)


def save_ecfp_family_landscape(families: pd.DataFrame, cfg: Config, out_path: Path) -> None:
    ensure_plot_dependencies()
    if PCA is None or families.empty or "ecfp_family" not in families.columns:
        return
    plot_df = families[families["ecfp_family"] != -1].copy()
    if plot_df.empty:
        return
    if len(plot_df) > cfg.max_pca_points:
        plot_df = plot_df.sample(cfg.max_pca_points, random_state=cfg.seed)
    fps, keep_rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            keep_rows.append(row)
    if len(fps) < 3:
        return
    embed_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    coords = PCA(n_components=2, random_state=cfg.seed).fit_transform(np.stack(fps))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]
    group_col = "ecfp_group" if "ecfp_group" in embed_df.columns else "ecfp_family"
    groups_sorted = sorted(int(g) for g in embed_df[group_col].dropna().unique() if int(g) != -1)
    group_counts = embed_df[group_col].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax = axes[0]
    cmap = plt.cm.tab10 if len(groups_sorted) <= 10 else plt.cm.tab20
    for i, group_id in enumerate(groups_sorted):
        sub = embed_df[embed_df[group_col].astype(int) == group_id]
        n_families = int(sub["ecfp_family"].nunique())
        label = f"G{group_id} ({len(sub)} mols, {n_families} fams)"
        ax.scatter(
            sub["x"], sub["y"], s=20, alpha=0.68,
            color=cmap(i / max(len(groups_sorted) - 1, 1)),
            label=label,
        )
    ax.set_title("Hierarchical ECFP/Tanimoto groups")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=7, ncol=1 if len(groups_sorted) <= 8 else 2)
    ax = axes[1]
    if "rank_score" in embed_df.columns:
        scores = pd.to_numeric(embed_df["rank_score"], errors="coerce")
        sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.72, c=scores, cmap="RdYlGn")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Ranking score")
    ax.set_title("Same projection colored by score")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle("Hierarchical ECFP chemical modules (sampled Butina families)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_ecfp_family_tree(run_dir: Path, out_path: Path) -> None:
    ensure_plot_dependencies()
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
    except Exception:
        return
    summary_path = run_dir / "ecfp_family_summary.csv"
    if not summary_path.exists():
        return
    try:
        summary = pd.read_csv(summary_path)
    except pd.errors.EmptyDataError:
        return
    if summary.empty or "representative_smiles" not in summary.columns:
        return
    top = summary.head(30).copy()
    fps, labels = [], []
    for row in top.itertuples(index=False):
        fp = morgan_fp(str(row.representative_smiles))
        if fp is not None:
            fps.append(fp)
            labels.append(f"F{int(row.ecfp_family)} n={int(row.n_molecules)}")
    if len(fps) < 3:
        return
    sim = compute_tanimoto_matrix(fps, fps)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    linked = linkage(condensed, method="average")
    fig, ax = plt.subplots(figsize=(12, max(5, len(labels) * 0.28)))
    dendrogram(linked, labels=labels, orientation="right", ax=ax, color_threshold=0.5)
    ax.set_xlabel("1 - Tanimoto similarity")
    ax.set_title("Hierarchy of top ECFP/Tanimoto families")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_score_summary_plot(
    summary_csv: Path,
    out_path: Path,
    id_col: str,
    title: str,
    label_prefix: str,
    max_rows: int = 16,
) -> None:
    ensure_plot_dependencies()
    if not summary_csv.exists():
        return
    try:
        df = pd.read_csv(summary_csv)
    except pd.errors.EmptyDataError:
        return
    if df.empty:
        return
    if "mean_score" not in df.columns and "mean_rank_score" in df.columns:
        df = df.rename(columns={"mean_rank_score": "mean_score"})
    if "max_score" not in df.columns and "max_rank_score" in df.columns:
        df = df.rename(columns={"max_rank_score": "max_score"})
    needed = {id_col, "n_molecules", "mean_score", "max_score"}
    if not needed.issubset(df.columns):
        return
    df["mean_score"] = pd.to_numeric(df["mean_score"], errors="coerce")
    df["max_score"] = pd.to_numeric(df["max_score"], errors="coerce")
    df["n_molecules"] = pd.to_numeric(df["n_molecules"], errors="coerce").fillna(0).astype(int)
    plot_df = df.dropna(subset=["mean_score", "max_score"]).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values(
        ["max_score", "mean_score", "n_molecules"],
        ascending=[False, False, False],
    ).head(max_rows)
    labels = [f"{label_prefix}{int(x)}\nn={n}" for x, n in zip(plot_df[id_col], plot_df["n_molecules"])]
    x = np.arange(len(plot_df))

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(plot_df) * 0.65), 5.6))
    ax = axes[0]
    ax.bar(x, plot_df["mean_score"], color="#4E79A7", alpha=0.82, label="mean")
    ax.scatter(x, plot_df["max_score"], color="#D62728", s=38, zorder=3, label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Ranking score")
    ax.set_title("Score by group")
    ax.legend(frameon=False)

    ax = axes[1]
    size = np.sqrt(plot_df["n_molecules"].clip(lower=1)) * 28
    sc = ax.scatter(
        plot_df["n_molecules"],
        plot_df["mean_score"],
        s=size,
        c=plot_df["max_score"],
        cmap="RdYlGn",
        alpha=0.78,
        edgecolor="black",
        linewidth=0.35,
    )
    for _, row in plot_df.head(10).iterrows():
        ax.text(row["n_molecules"], row["mean_score"], f"{label_prefix}{int(row[id_col])}", fontsize=8)
    ax.set_xlabel("Molecules in group")
    ax.set_ylabel("Mean ranking score")
    ax.set_title("Size vs score")
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Max ranking score")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def scaffold_family_label(family_id: int, family_smiles: list[str]) -> str:
    """Generate a short human-readable label for a scaffold family."""
    if family_id == -1:
        return "singleton"
    if not family_smiles:
        return f"F{family_id}"
    # Pick the most common scaffold in the family as the label
    from collections import Counter
    most_common = Counter(family_smiles).most_common(1)[0][0]
    # Truncate long SMILES
    label = most_common[:18] + "…" if len(most_common) > 18 else most_common
    return f"F{family_id}: {label}"


def save_scaffold_family_space(
    clustered: pd.DataFrame, cfg: Config, out_path: Path,
) -> None:
    """
    PCA of molecules colored by scaffold family.
    Reveals whether the chemical space clusters align with scaffold families,
    and which families are associated with high ranking scores.
    """
    ensure_plot_dependencies()
    if PCA is None or clustered.empty:
        return

    plot_df = clustered.copy()
    if len(plot_df) > cfg.max_pca_points:
        plot_df = plot_df.sample(cfg.max_pca_points, random_state=cfg.seed)

    fps, keep_rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            keep_rows.append(row)
    if not fps:
        return

    embed_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    coords = PCA(n_components=2, random_state=cfg.seed).fit_transform(np.stack(fps))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]

    families = sorted(embed_df["scaffold_family"].unique())
    n_real = len([f for f in families if f != -1])
    cmap = plt.cm.tab20 if n_real <= 20 else plt.cm.hsv

    has_score = "rank_score" in embed_df.columns and embed_df["rank_score"].notna().any()
    fig, axes = plt.subplots(1, 1 + int(has_score), figsize=(9 * (1 + int(has_score)), 8))
    if not has_score:
        axes = [axes]

    ax = axes[0]
    real_families = [f for f in families if f != -1]
    colors = {f: cmap(i / max(len(real_families), 1)) for i, f in enumerate(real_families)}
    colors[-1] = (0.8, 0.8, 0.8, 0.3)  # singletons: light grey, very transparent

    # Plot singletons first (background)
    noise = embed_df[embed_df["scaffold_family"] == -1]
    if not noise.empty:
        ax.scatter(noise["x"], noise["y"], s=12, alpha=0.2, color="lightgrey", label=f"singleton (n={len(noise)})")

    for fam in real_families:
        sub = embed_df[embed_df["scaffold_family"] == fam]
        scaffolds = sub["scaffold"].tolist() if "scaffold" in sub.columns else []
        label = scaffold_family_label(fam, scaffolds)
        ax.scatter(sub["x"], sub["y"], s=25, alpha=0.75, color=colors[fam], label=label)

    ax.set_title(f"Scaffold families (HDBSCAN, {n_real} families)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    if n_real <= 15:
        ax.legend(frameon=False, fontsize=7, ncol=2)

    if has_score:
        ax2 = axes[1]
        scores = embed_df["rank_score"].fillna(0).astype(float)
        span = scores.max() - scores.min()
        normalized = (scores - scores.min()) / span if span > 0 else scores * 0
        sizes = 10 + 40 * normalized
        sc = ax2.scatter(embed_df["x"], embed_df["y"], s=sizes, alpha=0.6,
                         c=scores, cmap="RdYlGn")
        plt.colorbar(sc, ax=ax2, shrink=0.8, label="Ranking score")
        ax2.set_title("Scaffold families sized and colored by score")
        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")

    fig.suptitle("Chemical space colored by scaffold family", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_scaffold_family_pocket_heatmap(
    clustered: pd.DataFrame,
    pocket_specs: list[PocketSpec],
    out_path: Path,
) -> None:
    """
    Heatmap: scaffold family × pocket — mean ranking score.

    Each cell shows the mean ranking score for molecules of that scaffold
    family when generated for that pocket. Reveals pocket-selective scaffold families.
    Also shows n_mols per cell as text annotation.
    """
    ensure_plot_dependencies()
    if clustered.empty:
        return

    has_score = "rank_score" in clustered.columns and clustered["rank_score"].notna().any()
    value_col = "rank_score" if has_score else "qed"

    scored = clustered[clustered["scaffold_family"] != -1].copy()
    if scored.empty:
        return

    # Expand pocket_ids (comma-joined) into one row per pocket
    rows = []
    for _, r in scored.iterrows():
        for pid in str(r.get("pocket_ids", "")).split(", "):
            pid = pid.strip()
            if pid:
                rows.append({**r.to_dict(), "pocket_id_single": pid})
    if not rows:
        return
    expanded = pd.DataFrame(rows)

    pivot_mean = expanded.groupby(["scaffold_family", "pocket_id_single"])[value_col].mean().unstack(fill_value=np.nan)
    pivot_count = expanded.groupby(["scaffold_family", "pocket_id_single"])[value_col].count().unstack(fill_value=0)

    # Order families by mean score descending
    family_order = pivot_mean.mean(axis=1).sort_values(ascending=False).index
    pivot_mean = pivot_mean.loc[family_order]
    pivot_count = pivot_count.loc[family_order]

    # Pocket column order matches pocket_specs order
    pocket_order = [s.pocket_id for s in pocket_specs if s.pocket_id in pivot_mean.columns]
    if pocket_order:
        pivot_mean = pivot_mean.reindex(columns=pocket_order, fill_value=np.nan)
        pivot_count = pivot_count.reindex(columns=pocket_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot_mean.columns) * 2),
                                    max(6, len(pivot_mean) * 0.5 + 2)))
    if value_col == "qed":
        im = ax.imshow(pivot_mean.values.astype(float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    else:
        im = ax.imshow(pivot_mean.values.astype(float), aspect="auto", cmap="RdYlGn")

    ax.set_xticks(range(len(pivot_mean.columns)))
    ax.set_xticklabels(pivot_mean.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot_mean.index)))

    # Family labels: family_id + example scaffold
    family_labels = []
    for fid in pivot_mean.index:
        sub = clustered[clustered["scaffold_family"] == fid]["scaffold"]
        example = sub.mode().iloc[0] if len(sub) else ""
        example = (example[:15] + "…") if len(example) > 15 else example
        family_labels.append(f"F{fid}: {example}")
    ax.set_yticklabels(family_labels, fontsize=8)

    # Annotate with count
    for i in range(len(pivot_mean.index)):
        for j in range(len(pivot_mean.columns)):
            cnt = int(pivot_count.values[i, j])
            val = pivot_mean.values[i, j]
            if cnt > 0:
                txt = f"{val:.2f}\n(n={cnt})"
                color = "black" if (np.isnan(val) or 0.3 < val < 0.7) else "white"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    label = "Mean ranking score" if has_score else "Mean QED"
    ax.set_title(f"Scaffold family × pocket — {label}\n"
                 f"(families ordered by mean score, only non-singleton families shown)")
    fig.colorbar(im, ax=ax, shrink=0.6, label=label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summary_dashboard(
    generated_df: pd.DataFrame, unique_df: pd.DataFrame, top_hits: pd.DataFrame,
    pocket_specs: list[PocketSpec], palette: dict[str, str], cfg: Config, out_path: Path,
) -> None:
    ensure_plot_dependencies()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    if generated_df.empty:
        for ax in axes.flatten():
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return
    counts = generated_df.groupby("generator")["smiles"].nunique().sort_values(ascending=False)
    ax1.bar(counts.index, counts.values, color=[palette.get(k, "#4E79A7") for k in counts.index])
    ax1.set_title("Unique molecules per generator")
    ax1.tick_params(axis="x", rotation=35)
    scaffolds = (
        generated_df.loc[generated_df["scaffold"] != ""]
        .groupby("generator")["scaffold"].nunique()
        .reindex(counts.index).fillna(0)
    )
    ax2.bar(scaffolds.index, scaffolds.values, color=[palette.get(k, "#4E79A7") for k in scaffolds.index])
    ax2.set_title("Unique scaffolds per generator")
    ax2.tick_params(axis="x", rotation=35)
    for gen in counts.index:
        sub = generated_df[generated_df["generator"] == gen]
        ax3.scatter(sub["mw"], sub["qed"], s=28, alpha=0.7, color=palette.get(gen, "#4E79A7"), label=gen)
    ax3.set_title("Property landscape")
    ax3.set_xlabel("Molecular weight")
    ax3.set_ylabel("QED")
    ax3.legend(frameon=False)
    ax4.axis("off")
    text_lines = [
        f"Target: {cfg.target_name}",
        f"PDB: {cfg.pdb_id}",
        f"Pockets: {len(pocket_specs)}",
        f"Generated rows: {len(generated_df):,}",
        f"Unique molecules: {unique_df['smiles'].nunique():,}",
    ]
    if not top_hits.empty:
        best = top_hits.iloc[0]
        score_col = "rank_score" if pd.notna(best.get("rank_score")) else None
        text_lines += ["", "Top hit:", best["smiles"],
                       f"Generators: {best.get('generators', '')}"]
        if score_col:
            text_lines.append(f"Score: {best.get(score_col, 0):.3f}")
    ax4.text(0.02, 0.98, "\n".join(text_lines), va="top", ha="left", family="monospace")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_pocket_generator_heatmap(generated_df: pd.DataFrame, out_path: Path) -> None:
    ensure_plot_dependencies()
    fig, ax = plt.subplots(figsize=(10, 6))
    if generated_df.empty:
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return
    pivot = generated_df.groupby(["pocket_id", "generator"])["qed"].max().unstack(fill_value=0)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Best QED per (pocket, generator)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_yield_matrix(generated_df: pd.DataFrame, out_path: Path) -> None:
    ensure_plot_dependencies()
    fig, ax = plt.subplots(figsize=(10, 6))
    if generated_df.empty:
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return
    pivot = generated_df.groupby(["pocket_id", "generator"])["smiles"].nunique().unstack(fill_value=0)
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, str(pivot.values[i, j]), ha="center", va="center", fontsize=9)
    ax.set_title("Unique molecules per (pocket, generator)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_scaffold_overlap(generated_df: pd.DataFrame, out_path: Path) -> None:
    ensure_plot_dependencies()
    fig, ax = plt.subplots(figsize=(6, 5))
    generators = sorted(generated_df["generator"].unique()) if not generated_df.empty else []
    if not generators:
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return
    scaffold_sets = {
        gen: set(generated_df.loc[
            (generated_df["generator"] == gen) & (generated_df["scaffold"] != ""), "scaffold"
        ])
        for gen in generators
    }
    mat = np.zeros((len(generators), len(generators)), dtype=float)
    for i, g1 in enumerate(generators):
        for j, g2 in enumerate(generators):
            union = scaffold_sets[g1] | scaffold_sets[g2]
            inter = scaffold_sets[g1] & scaffold_sets[g2]
            mat[i, j] = len(inter) / len(union) if union else 0.0
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_xticks(range(len(generators)))
    ax.set_xticklabels(generators, rotation=45, ha="right")
    ax.set_yticks(range(len(generators)))
    ax.set_yticklabels(generators)
    ax.set_title("Scaffold Jaccard overlap between generators")
    for i in range(len(generators)):
        for j in range(len(generators)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_pocket_druggability(
    unique_df: pd.DataFrame, pocket_specs: list[PocketSpec], out_path: Path,
) -> None:
    ensure_plot_dependencies()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    if not pocket_specs:
        ax1.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return
    pocket_ids = [s.pocket_id for s in pocket_specs]
    p2rank_scores = [s.p2rank_score for s in pocket_specs]
    best_scores = []
    for pid in pocket_ids:
        pocket_mols = unique_df[unique_df["pocket_ids"].str.contains(pid, na=False)]
        if pocket_mols.empty:
            best_scores.append(0.0)
        elif "rank_score" in pocket_mols and pocket_mols["rank_score"].notna().any():
            best_scores.append(float(pocket_mols["rank_score"].max()))
        else:
            best_scores.append(0.0)
    x = list(range(len(pocket_ids)))
    ax1.bar(x, best_scores, color="#4E79A7", alpha=0.8, label="Best mol score")
    ax1.set_ylabel("Best molecule score")
    ax1.set_title("Pocket druggability vs molecule quality")
    ax2 = ax1.twinx()
    ax2.plot(x, p2rank_scores, "ro-", label="P2Rank score")
    ax2.set_ylabel("P2Rank druggability")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pocket_ids, rotation=45, ha="right")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

# Analysis rebuild helpers
# ---------------------------------------------------------------------------

def build_generator_palette(generator_names: list[str]) -> dict[str, str]:
    base = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948"]
    labels = sorted({display_generator_name(name) for name in generator_names})
    return {name: base[i % len(base)] for i, name in enumerate(labels)}


def write_standard_analysis_csvs(run_dir: Path, generated_df: pd.DataFrame) -> None:
    try:
        build_scaffold_diversity_matrix(generated_df).to_csv(run_dir / "scaffold_diversity_matrix.csv")
        pocket_sensitivity(generated_df).to_csv(run_dir / "generator_pocket_sensitivity.csv", header=True)
        compute_budget_summary(generated_df).to_csv(run_dir / "compute_budget_summary.csv")
        scaffold_df = generated_df[generated_df["scaffold"] != ""]
        for pocket_id in scaffold_df["pocket_id"].unique():
            pocket_jaccard_matrix(
                scaffold_df[scaffold_df["pocket_id"] == pocket_id]
            ).to_csv(run_dir / f"scaffold_jaccard_{pocket_id}.csv")
    except Exception as exc:
        print(f"WARNING: analysis step failed: {exc}", flush=True)

    scaffold_df = generated_df.loc[generated_df["scaffold"] != ""]
    if not scaffold_df.empty:
        scaffold_df.groupby(["scaffold", "generator", "pocket_id"]).size().unstack(
            fill_value=0
        ).to_csv(run_dir / "scaffold_presence.csv")


def write_presentation_analysis_csvs(run_dir: Path, merged: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    try:
        clustered = cluster_molecule_fingerprints(
            merged,
            max_points=cfg.max_cluster_points,
            seed=cfg.seed,
        )
        write_cluster_summary(clustered, run_dir / "cluster_summary.csv")
        write_top_hit_enrichment(
            clustered, "generators", run_dir / "top_hit_enrichment_by_generator.csv", "generator"
        )
        write_top_hit_enrichment(
            clustered, "pocket_ids", run_dir / "top_hit_enrichment_by_pocket.csv", "pocket"
        )
        write_top_hit_enrichment(
            clustered, "chemical_cluster", run_dir / "top_hit_enrichment_by_cluster.csv", "cluster"
        )
        return clustered
    except Exception as exc:
        print(f"WARNING: presentation analysis failed: {exc}", flush=True)
        return merged.copy()
