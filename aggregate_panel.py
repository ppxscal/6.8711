"""Aggregate cross-target tables and figures for the Pocket-Aware Ensemble Audit paper.

Reads completed results/panel_*_<target>/ directories, collates per-target CSVs,
and writes headline tables (paper/tables/) and main-text figures (paper/figures/).

Usage:
    python aggregate_panel.py
    python aggregate_panel.py --targets brd4_bd1 btk ggct stk33 vhl
    python aggregate_panel.py --results-glob 'results/panel_20260503_*'
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DEFAULT_GLOB = "results/panel_20260503_*"
EXCLUDE_SUFFIXES = ("_old",)
OUT_TABLES = ROOT / "paper" / "tables"
OUT_FIGS = ROOT / "paper" / "figures"
GENERATOR_ORDER = ["DiffSBDD", "PocketXMol", "PocketXMolAR"]
GENERATOR_COLOR = {
    "DiffSBDD": "#4C72B0",
    "PocketXMol": "#DD8452",
    "PocketXMolAR": "#55A868",
}


def target_name(run_dir: Path) -> str:
    m = re.match(r"panel_\d+_(.+)", run_dir.name)
    return m.group(1) if m else run_dir.name


def discover_targets(glob: str, include: list[str] | None) -> list[Path]:
    dirs = sorted(ROOT.glob(glob))
    keep = []
    for d in dirs:
        if not d.is_dir():
            continue
        name = target_name(d)
        if any(name.endswith(s) for s in EXCLUDE_SUFFIXES):
            continue
        if include and name not in include:
            continue
        if not (d / "scored_candidates_rtmscore.csv").exists():
            continue
        keep.append(d)
    return keep


def load_scored(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "scored_candidates_rtmscore.csv")
    df["target"] = target_name(run_dir)
    return df


def aggregate_predictability(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        p = d / "source_pocket_predictability.csv"
        if not p.exists() or p.stat().st_size < 2:
            continue
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df["target"] = target_name(d)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    merged = pd.concat(rows, ignore_index=True)
    merged = merged.rename(columns={"subset": "generator"})
    return merged


def summary_predictability(pred: pd.DataFrame) -> pd.DataFrame:
    gen = pred[pred["generator"].isin(GENERATOR_ORDER)].copy()
    summary = (
        gen.groupby("generator")
        .agg(
            n_targets=("target", "nunique"),
            mean_balanced_accuracy=("balanced_accuracy_mean", "mean"),
            mean_shuffled=("shuffled_balanced_accuracy_mean", "mean"),
            mean_delta=("delta_vs_shuffled", "mean"),
            std_delta=("delta_vs_shuffled", "std"),
            min_delta=("delta_vs_shuffled", "min"),
            max_delta=("delta_vs_shuffled", "max"),
        )
        .reindex(GENERATOR_ORDER)
    )
    return summary.reset_index()


def aggregate_score_summary(run_dirs: list[Path]) -> pd.DataFrame:
    """Returns one row per unique molecule: best-pose RTMScore + scaffold + RA score."""
    rows = []
    for d in run_dirs:
        target = target_name(d)
        scored_path = d / "scored_candidates_rtmscore.csv"
        if not scored_path.exists():
            continue
        df = pd.read_csv(scored_path)
        ra_path = d / "ra_scores.csv"
        if ra_path.exists():
            ra = pd.read_csv(ra_path)
            df = df.merge(ra, on="smiles", how="left")
        df = df.rename(columns={
            "rtmscore_best_generator": "generator",
            "rtmscore_best_pocket_id": "pocket_id",
            "rtmscore_score": "score",
        })
        df = df[df["generator"].isin(GENERATOR_ORDER)].copy()
        df["target"] = target
        keep = ["target", "generator", "pocket_id", "smiles", "scaffold",
                "score", "qed", "mw", "logp"]
        if "ra_score" in df.columns:
            keep.append("ra_score")
        rows.append(df[keep])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def top_fraction_jaccard(scored: pd.DataFrame, frac: float = 0.05,
                         key: str = "scaffold") -> pd.DataFrame:
    """For each target, compute pairwise Jaccard on top-`frac` sets across generators,
    grouped by `key` (default Bemis-Murcko scaffold, since exact SMILES never overlap)."""
    rows = []
    for target, sub in scored.groupby("target"):
        sub = sub.dropna(subset=[key])
        sub = sub[sub[key].astype(str).str.len() > 0]
        if len(sub) == 0:
            continue
        k = max(1, int(np.ceil(len(sub) * frac)))
        threshold = sub["score"].nlargest(k).min()
        top = sub[sub["score"] >= threshold]
        top_sets = {g: set(top[top["generator"] == g][key]) for g in GENERATOR_ORDER}
        for i, gi in enumerate(GENERATOR_ORDER):
            for gj in GENERATOR_ORDER[i + 1 :]:
                a, b = top_sets.get(gi, set()), top_sets.get(gj, set())
                union = a | b
                jacc = len(a & b) / len(union) if union else 0.0
                rows.append({"target": target, "gen_a": gi, "gen_b": gj, "jaccard": jacc,
                             "n_a": len(a), "n_b": len(b), "n_intersect": len(a & b),
                             "top_k": k, "key": key})
    return pd.DataFrame(rows)


def top_fraction_enrichment(scored: pd.DataFrame, frac: float = 0.05) -> pd.DataFrame:
    rows = []
    for target, sub in scored.groupby("target"):
        k = max(1, int(np.ceil(len(sub) * frac)))
        threshold = sub["score"].nlargest(k).min()
        top = sub[sub["score"] >= threshold]
        counts = top["generator"].value_counts()
        totals = sub["generator"].value_counts()
        for g in GENERATOR_ORDER:
            n_top = int(counts.get(g, 0))
            n_tot = int(totals.get(g, 0))
            rows.append({
                "target": target, "generator": g,
                "n_top": n_top, "n_total": n_tot,
                "fraction_of_top": n_top / k if k else 0.0,
                "within_gen_rate": n_top / n_tot if n_tot else 0.0,
                "top_k": k,
            })
    return pd.DataFrame(rows)


def pocket_difficulty(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, pocket), sub in scored.groupby(["target", "pocket_id"]):
        rows.append({
            "target": target, "pocket_id": pocket,
            "n": len(sub),
            "mean_score": sub["score"].mean(),
            "median_score": sub["score"].median(),
            "p95_score": sub["score"].quantile(0.95),
        })
    return pd.DataFrame(rows)


# -------------------- figures --------------------

def fig_predictability(summary: pd.DataFrame, pred: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    x = np.arange(len(GENERATOR_ORDER))
    means = summary.set_index("generator").reindex(GENERATOR_ORDER)["mean_delta"].values
    stds = summary.set_index("generator").reindex(GENERATOR_ORDER)["std_delta"].fillna(0).values
    colors = [GENERATOR_COLOR[g] for g in GENERATOR_ORDER]
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=4,
           edgecolor="black", linewidth=0.6)
    # Overlay per-target points.
    per_gen = pred[pred["generator"].isin(GENERATOR_ORDER)]
    for i, g in enumerate(GENERATOR_ORDER):
        vals = per_gen[per_gen["generator"] == g]["delta_vs_shuffled"].values
        ax.scatter(np.full_like(vals, i, dtype=float) + np.random.uniform(-0.08, 0.08, len(vals)),
                   vals, color="black", s=18, zorder=3, alpha=0.7)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(GENERATOR_ORDER)
    ax.set_ylabel("Δ balanced accuracy vs. label shuffle")
    ax.set_title("Pocket-conditioning strength by generator\n(higher = outputs reveal which pocket they came from)")
    ax.set_ylim(bottom=min(0, (means - stds).min() - 0.05))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def fig_score_boxplots(scored: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    # by generator
    data = [scored[scored["generator"] == g]["score"].values for g in GENERATOR_ORDER]
    bp = axes[0].boxplot(data, labels=GENERATOR_ORDER, patch_artist=True, showfliers=False)
    for patch, g in zip(bp["boxes"], GENERATOR_ORDER):
        patch.set_facecolor(GENERATOR_COLOR[g])
        patch.set_alpha(0.7)
    axes[0].set_ylabel("RTMScore (best pose per molecule)")
    axes[0].set_title("Score by generator (all targets pooled)")
    # by target
    targets = sorted(scored["target"].unique())
    data_t = [scored[scored["target"] == t]["score"].values for t in targets]
    bp2 = axes[1].boxplot(data_t, labels=targets, patch_artist=True, showfliers=False)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#BBBBBB")
    axes[1].set_title("Score by target")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def fig_top_enrichment(enrich: pd.DataFrame, out_path: Path, frac: float = 0.05) -> None:
    pivot = enrich.pivot(index="target", columns="generator", values="fraction_of_top").reindex(columns=GENERATOR_ORDER)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    pivot.plot(kind="bar", ax=ax, color=[GENERATOR_COLOR[g] for g in GENERATOR_ORDER],
               edgecolor="black", linewidth=0.5)
    ax.axhline(1 / len(GENERATOR_ORDER), color="grey", ls="--", lw=0.8, label="equal share")
    ax.set_ylabel(f"Fraction of top-{int(frac*100)}% hits")
    ax.set_title(f"Generator share of top-{int(frac*100)}% RTMScore (per target)")
    ax.legend(title=None, fontsize=8)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def fig_score_vs_ra(scored: pd.DataFrame, out_path: Path) -> None:
    if "ra_score" not in scored.columns:
        return
    sub = scored.dropna(subset=["ra_score", "score"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    # Scatter of all mols
    for g in GENERATOR_ORDER:
        gs = sub[sub["generator"] == g]
        axes[0].scatter(gs["ra_score"], gs["score"], s=4, alpha=0.25,
                        color=GENERATOR_COLOR[g], label=g)
    axes[0].set_xlabel("RA score (higher = easier to synthesize)")
    axes[0].set_ylabel("RTMScore")
    axes[0].set_title("Score vs. synthesizability (all mols)")
    axes[0].legend(markerscale=2, fontsize=8)
    # Top-5% RA distribution per generator
    rows = []
    for target, grp in sub.groupby("target"):
        k = max(1, int(np.ceil(len(grp) * 0.05)))
        thr = grp["score"].nlargest(k).min()
        top = grp[grp["score"] >= thr]
        for g in GENERATOR_ORDER:
            vals = top[top["generator"] == g]["ra_score"].values
            for v in vals:
                rows.append({"generator": g, "ra_score": v, "target": target})
    top_df = pd.DataFrame(rows)
    if not top_df.empty:
        data = [top_df[top_df["generator"] == g]["ra_score"].values for g in GENERATOR_ORDER]
        bp = axes[1].boxplot(data, tick_labels=GENERATOR_ORDER, patch_artist=True, showfliers=False)
        for patch, g in zip(bp["boxes"], GENERATOR_ORDER):
            patch.set_facecolor(GENERATOR_COLOR[g]); patch.set_alpha(0.7)
        axes[1].set_ylabel("RA score among top-5% scorers")
        axes[1].set_title("Are top hits synthesizable?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def per_target_generator_variance(scored: pd.DataFrame) -> pd.DataFrame:
    """Mean + std of score by (target, generator) to show DiffSBDD variance."""
    rows = []
    for (target, g), sub in scored.groupby(["target", "generator"]):
        rows.append({
            "target": target, "generator": g, "n": len(sub),
            "mean_score": sub["score"].mean(),
            "std_score": sub["score"].std(),
            "p95_score": sub["score"].quantile(0.95),
            "p5_score": sub["score"].quantile(0.05),
        })
    return pd.DataFrame(rows)


def fig_jaccard(jacc: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    jacc["pair"] = jacc["gen_a"] + " ↔ " + jacc["gen_b"]
    pivot = jacc.pivot(index="target", columns="pair", values="jaccard")
    pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Jaccard on top-5% molecules")
    ax.set_title("Top-5% overlap across generators (per target)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(0.1, jacc["jaccard"].max() * 1.2))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-glob", default=DEFAULT_GLOB)
    parser.add_argument("--targets", nargs="*", default=None)
    args = parser.parse_args()

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_targets(args.results_glob, args.targets)
    if not run_dirs:
        raise SystemExit("No scored target directories found.")
    print(f"Using {len(run_dirs)} targets: {[target_name(d) for d in run_dirs]}")

    # --- Predictability ---
    pred = aggregate_predictability(run_dirs)
    pred.to_csv(OUT_TABLES / "predictability_per_target.csv", index=False)
    pred_summary = summary_predictability(pred)
    pred_summary.to_csv(OUT_TABLES / "predictability_summary.csv", index=False)
    print("\n=== Pocket-conditioning strength (Δ balanced accuracy vs shuffle) ===")
    print(pred_summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- Scores ---
    scored = aggregate_score_summary(run_dirs)
    if not scored.empty:
        scored.to_csv(OUT_TABLES / "scored_best_pose_per_mol.csv", index=False)
        per_target_gen = (
            scored.groupby(["target", "generator"])["score"]
            .agg(["count", "mean", "median", lambda s: s.quantile(0.95)])
            .rename(columns={"<lambda_0>": "p95"})
            .reset_index()
        )
        per_target_gen.to_csv(OUT_TABLES / "score_summary_by_target_generator.csv", index=False)
        enrich = top_fraction_enrichment(scored, frac=0.05)
        enrich.to_csv(OUT_TABLES / "top5pct_enrichment.csv", index=False)
        jacc = top_fraction_jaccard(scored, frac=0.05)
        jacc.to_csv(OUT_TABLES / "top5pct_jaccard.csv", index=False)
        diff = pocket_difficulty(scored)
        diff.to_csv(OUT_TABLES / "pocket_difficulty.csv", index=False)
        variance = per_target_generator_variance(scored)
        variance.to_csv(OUT_TABLES / "score_variance_by_target_generator.csv", index=False)

        print("\n=== Top-5% share by (target, generator) ===")
        print(enrich.pivot(index="target", columns="generator", values="fraction_of_top")
              .reindex(columns=GENERATOR_ORDER).round(3).to_string())

        print("\n=== Top-5% Jaccard across generators ===")
        print(jacc.groupby(["gen_a", "gen_b"])["jaccard"].mean().round(4).to_string())

    # --- Figures ---
    if not pred.empty:
        fig_predictability(pred_summary, pred, OUT_FIGS / "fig1_predictability.png")
    if not scored.empty:
        fig_score_boxplots(scored, OUT_FIGS / "fig2_score_boxplots.png")
        fig_top_enrichment(enrich, OUT_FIGS / "fig3_top5_enrichment.png")
        fig_jaccard(jacc, OUT_FIGS / "fig4_top5_jaccard.png")
        fig_score_vs_ra(scored, OUT_FIGS / "fig5_score_vs_ra.png")

    print(f"\nWrote tables to {OUT_TABLES} and figures to {OUT_FIGS}")


if __name__ == "__main__":
    main()
