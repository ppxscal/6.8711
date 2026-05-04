"""Does the mutual / overlapping region of generators in fingerprint space
have systematically higher RTMScore or RA than a generator's exclusive region?

Operationalization:
 - For each target, compute Morgan fingerprints for all unique molecules.
 - For each molecule, find its k nearest neighbors (in Tanimoto or fingerprint-PCA Euclidean).
 - Compute Shannon entropy of the neighbors' generator identity. High entropy ⇒ molecule
   sits where multiple generators overlap; low entropy ⇒ monoculture region.
 - Correlate per-molecule generator-entropy with RTMScore and RA score.

Writes paper/tables/overlap_entropy_correlations.csv and a headline figure.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent
RESULTS_GLOB = "results/panel_20260503_*"
EXCLUDE_SUFFIXES = ("_old",)
OUT_TABLE = ROOT / "paper" / "tables" / "overlap_entropy_correlations.csv"
OUT_FIG = ROOT / "paper" / "figures" / "fig7_overlap_entropy.png"
GENERATOR_ORDER = ["DiffSBDD", "PocketXMol", "PocketXMolAR"]
GENERATOR_COLOR = {"DiffSBDD": "#4C72B0", "PocketXMol": "#DD8452", "PocketXMolAR": "#55A868"}

K_NEIGHBORS = 20
FP_BITS = 2048
FP_RADIUS = 2
MAX_MOLS_PER_TARGET = 6000   # cap for compute; sample if target is huge


def morgan_fp(smi: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    arr = np.zeros(FP_BITS, dtype=np.uint8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(bv, arr)
    return arr


def entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def analyze_target(run_dir: Path) -> pd.DataFrame:
    target = run_dir.name.replace("panel_20260503_", "")
    gen = pd.read_csv(run_dir / "generated_by_generator.csv")
    ra = pd.read_csv(run_dir / "ra_scores.csv")
    scored = pd.read_csv(run_dir / "scored_candidates_rtmscore.csv")

    # unique smi -> generator (primary) + scores
    gen_primary = (
        gen.groupby("smiles")["generator"].first().reset_index()
    )
    df = gen_primary.merge(ra, on="smiles", how="left")
    df = df.merge(
        scored[["smiles", "rtmscore_score", "rtmscore_best_pocket_id"]],
        on="smiles", how="left"
    )
    df = df.dropna(subset=["rtmscore_score", "ra_score"])
    df = df[df["generator"].isin(GENERATOR_ORDER)].reset_index(drop=True)

    if len(df) > MAX_MOLS_PER_TARGET:
        df = df.sample(MAX_MOLS_PER_TARGET, random_state=0).reset_index(drop=True)

    # fingerprints
    fps = []
    keep_idx = []
    for i, smi in enumerate(df["smiles"].values):
        fp = morgan_fp(smi)
        if fp is None:
            continue
        fps.append(fp)
        keep_idx.append(i)
    df = df.iloc[keep_idx].reset_index(drop=True)
    X = np.vstack(fps).astype(np.float32)

    # Jaccard ~ 1 - Tanimoto; use sklearn "jaccard" metric on binary vectors
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric="jaccard",
                            algorithm="brute", n_jobs=-1)
    nbrs.fit(X)
    _, idx = nbrs.kneighbors(X)
    # drop self-neighbor
    idx = idx[:, 1:]

    gen_arr = df["generator"].values
    gen_to_i = {g: i for i, g in enumerate(GENERATOR_ORDER)}
    entropies = np.zeros(len(df))
    own_frac = np.zeros(len(df))
    for i, nb in enumerate(idx):
        counts = np.zeros(len(GENERATOR_ORDER))
        for j in nb:
            counts[gen_to_i[gen_arr[j]]] += 1
        entropies[i] = entropy(counts)
        own_frac[i] = counts[gen_to_i[gen_arr[i]]] / K_NEIGHBORS

    df["neighbor_entropy"] = entropies
    df["own_generator_frac"] = own_frac
    df["target"] = target
    return df[["target", "smiles", "generator", "rtmscore_score", "ra_score",
               "neighbor_entropy", "own_generator_frac"]]


def main() -> None:
    dirs = []
    for d in sorted(ROOT.glob(RESULTS_GLOB)):
        if not d.is_dir():
            continue
        name = d.name.replace("panel_20260503_", "")
        if any(name.endswith(s) for s in EXCLUDE_SUFFIXES):
            continue
        if (d / "scored_candidates_rtmscore.csv").exists() and (d / "ra_scores.csv").exists():
            dirs.append(d)

    frames = []
    for d in dirs:
        print(f"Processing {d.name} ...", flush=True)
        frames.append(analyze_target(d))
    pooled = pd.concat(frames, ignore_index=True)
    pooled.to_csv(ROOT / "paper" / "tables" / "overlap_entropy_per_molecule.csv", index=False)

    # Correlations: per-target and pooled, overall and per-generator.
    rows = []
    for (target, g), sub in pooled.groupby(["target", "generator"]):
        if len(sub) < 20:
            continue
        for score_col in ["rtmscore_score", "ra_score"]:
            rho, p = spearmanr(sub["neighbor_entropy"], sub[score_col])
            rows.append({
                "scope": "per_target_per_gen", "target": target, "generator": g,
                "score": score_col, "n": len(sub), "spearman_rho": rho, "p": p,
            })

    for g, sub in pooled.groupby("generator"):
        for score_col in ["rtmscore_score", "ra_score"]:
            rho, p = spearmanr(sub["neighbor_entropy"], sub[score_col])
            rows.append({
                "scope": "pooled_per_gen", "target": "ALL", "generator": g,
                "score": score_col, "n": len(sub), "spearman_rho": rho, "p": p,
            })

    for score_col in ["rtmscore_score", "ra_score"]:
        rho, p = spearmanr(pooled["neighbor_entropy"], pooled[score_col])
        rows.append({
            "scope": "pooled_all", "target": "ALL", "generator": "ALL",
            "score": score_col, "n": len(pooled), "spearman_rho": rho, "p": p,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLE, index=False)

    print("\n=== Overlap-region entropy correlations (Spearman) ===")
    piv = out[out["scope"] == "pooled_per_gen"].pivot(
        index="generator", columns="score", values="spearman_rho"
    ).reindex(GENERATOR_ORDER).round(3)
    print("Per-generator pooled across targets:\n", piv, sep="")

    overall = out[out["scope"] == "pooled_all"].set_index("score")["spearman_rho"].round(3)
    print("\nOverall pooled (all generators, all targets):")
    print(overall)

    # Figure: binned score vs neighbor_entropy, per generator.
    # Use fixed bin edges on [0, ln(3)] so x-axes match across panels and
    # within-bin IQR shading to show dispersion (so the RA panel doesn't look spiky).
    ln3 = float(np.log(3))
    edges = np.linspace(0.0, ln3 * 1.001, 9)  # 8 bins across the full theoretical range
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for i, score_col in enumerate(["rtmscore_score", "ra_score"]):
        ax = axes[i]
        # Offset generator markers horizontally within each bin so error bars don't overlap.
        bin_width = edges[1] - edges[0]
        offsets = np.linspace(-0.18, 0.18, len(GENERATOR_ORDER)) * bin_width
        for gi, g in enumerate(GENERATOR_ORDER):
            sub = pooled[pooled["generator"] == g].copy()
            if sub.empty:
                continue
            sub["bin"] = pd.cut(sub["neighbor_entropy"], edges, include_lowest=True)
            grp = sub.groupby("bin", observed=True)[score_col]
            med = grp.median()
            counts = grp.size()
            # Distribution-free 95% CI for the median (order statistics, normal approx).
            # Band i = ceil(n/2 - 1.96 sqrt(n)/2), j = ceil(n/2 + 1.96 sqrt(n)/2)
            ci_lo, ci_hi = [], []
            for b, n in counts.items():
                vals = np.sort(sub.loc[sub["bin"] == b, score_col].values)
                if n < 30:
                    ci_lo.append(np.nan); ci_hi.append(np.nan); continue
                half = 1.96 * np.sqrt(n) / 2.0
                lo_idx = max(0, int(np.ceil(n / 2 - half)) - 1)
                hi_idx = min(n - 1, int(np.ceil(n / 2 + half)) - 1)
                ci_lo.append(vals[lo_idx]); ci_hi.append(vals[hi_idx])
            ci_lo = np.array(ci_lo); ci_hi = np.array(ci_hi)
            keep = counts.values >= 30
            x = np.array(centers)[keep] + offsets[gi]
            y = med.values[keep]
            lo_err = y - ci_lo[keep]
            hi_err = ci_hi[keep] - y
            if len(x) < 2:
                continue
            ax.errorbar(x, y, yerr=[lo_err, hi_err],
                        fmt="o-", color=GENERATOR_COLOR[g], label=g,
                        linewidth=1.5, markersize=5,
                        capsize=3, elinewidth=1.2, capthick=1.2)
        if i == 0:
            ax.set_ylabel("Median RTMScore")
            ax.set_title("RTMScore vs. overlap depth")
        else:
            ax.set_ylabel("Median RA score")
            ax.set_title("Retrosynthesis accessibility vs. overlap depth")
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="best")
        ax.set_xlim(0, ln3)

    # Single shared x-label at the bottom of the figure.
    fig.supxlabel(
        "Neighbor-entropy over generator identity "
        "(0 = molecule sits in its own generator's monoculture, "
        f"ln 3 ≈ {ln3:.2f} = neighbors are fully mixed across generators)",
        fontsize=9, y=-0.02,
    )
    fig.suptitle(
        "Do molecules in the cross-generator overlap region score higher?",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_FIG.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"\nTable:  {OUT_TABLE}")
    print(f"Figure: {OUT_FIG}")


if __name__ == "__main__":
    main()
