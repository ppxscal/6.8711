#!/usr/bin/env bash
set -euo pipefail

CHORUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${CHORUS_ROOT:-$CHORUS_DIR}"
ENV_ROOT="${ENV_ROOT:-$REPO_ROOT/envs/uv}"
ENVS_DIR="${ENVS_DIR:-$(dirname "$ENV_ROOT")}"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$REPO_ROOT/checkpoints}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CHORUS_DIR/cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

RUN_NAME="${RUN_NAME:-}"
REFRESH="${REFRESH:-false}"
SCORER="${SCORER:-rtmscore}"
GPU_DEVICES="${GPU_DEVICES:-}"
MAX_GPUS="${MAX_GPUS:-}"

scored_cache_name() {
    local scorer
    scorer="$(printf "%s" "$1" | tr '[:upper:]' '[:lower:]')"
    echo "scored_candidates_${scorer}.csv"
}

cached_analysis_ready() {
    if [[ -z "$RUN_NAME" || "$REFRESH" == "true" ]]; then
        return 1
    fi
    local run_dir="$CHORUS_DIR/results/$RUN_NAME"
    local scored_name
    scored_name="$(scored_cache_name "$SCORER")"
    [[ -f "$run_dir/generated_by_generator.csv" && \
       -f "$run_dir/unique_generated.csv" && \
       -f "$run_dir/$scored_name" ]]
}

find_python() {
    local imports="import hdbscan, matplotlib, numpy, pandas, rdkit, sklearn, yaml"
    local purpose="full pipeline"
    if cached_analysis_ready; then
        imports="import hdbscan, matplotlib, numpy, pandas, rdkit, sklearn, yaml"
        purpose="cached analysis"
    fi
    for candidate in \
        "$ENV_ROOT/chorus/bin/python" \
        "$ENV_ROOT/diffsbdd/bin/python" \
        "$REPO_ROOT/unicellular/.venv/bin/python" \
        "$(which python3 2>/dev/null)"; do
        if [[ -x "$candidate" ]] && "$candidate" -c "$imports" 2>/dev/null; then
            echo "$candidate"
            return 0
        fi
    done
    echo "Error: could not find a Python for $purpose with required packages." >&2
    if cached_analysis_ready; then
        echo "Run: bash setup.sh chorus" >&2
    else
        echo "Run: bash setup.sh all" >&2
    fi
    return 1
}
PYTHON="$(find_python)"

if [[ -z "${PDB_ID:-}" ]]; then
    echo "Error: PDB_ID is required."
    echo "Usage: PDB_ID=2PN7 TARGET_NAME=GGCT bash run.sh"
    exit 1
fi

TARGET_NAME="${TARGET_NAME:-$PDB_ID}"
N_PER_POCKET="${N_PER_POCKET:-1250}"
MODE="${MODE:-real}"
QUIET="${QUIET:-true}"
GENERATORS="${GENERATORS:-}"
MAX_PCA_POINTS="${MAX_PCA_POINTS:-5000}"
MAX_UMAP_POINTS="${MAX_UMAP_POINTS:-1000}"
MAX_CLUSTER_POINTS="${MAX_CLUSTER_POINTS:-1000}"
ECFP_FAMILY_SIM_THRESHOLD="${ECFP_FAMILY_SIM_THRESHOLD:-0.30}"
MAX_TANIMOTO_REFS_PER_POCKET="${MAX_TANIMOTO_REFS_PER_POCKET:-500}"
TANIMOTO_TOP_K="${TANIMOTO_TOP_K:-10}"
P2RANK_MAX_POCKETS="${P2RANK_MAX_POCKETS:-4}"
P2RANK_MIN_SCORE="${P2RANK_MIN_SCORE:-0.5}"
POCKET_RADIUS="${POCKET_RADIUS:-10.0}"
POCKET_BBOX_SIZE="${POCKET_BBOX_SIZE:-23.0}"
LIGAND_RESNAME="${LIGAND_RESNAME:-}"
ANCHOR_RESIDUE="${ANCHOR_RESIDUE:-}"

echo "=== Chorus Run ==="
echo "PDB:         $PDB_ID"
echo "Target name: $TARGET_NAME"
echo "N/pocket:    $N_PER_POCKET"
echo "Mode:        $MODE"
echo "Max pockets: $P2RANK_MAX_POCKETS"
[[ -n "$RUN_NAME" ]] && echo "Run name:    $RUN_NAME"
[[ -n "$GENERATORS" ]] && echo "Generators:  $GENERATORS"
echo "Scorer:      $SCORER"
[[ -n "$GPU_DEVICES" ]] && echo "GPU devices: $GPU_DEVICES"
[[ -n "$MAX_GPUS" ]] && echo "Max GPUs:    $MAX_GPUS"
echo "Analysis cap: PCA=$MAX_PCA_POINTS UMAP=$MAX_UMAP_POINTS cluster=$MAX_CLUSTER_POINTS"
echo "ECFP family: sim_threshold=$ECFP_FAMILY_SIM_THRESHOLD"
echo "Tanimoto:    refs/pocket=$MAX_TANIMOTO_REFS_PER_POCKET top_k=$TANIMOTO_TOP_K"
[[ -n "$LIGAND_RESNAME" ]] && echo "Ligand:      $LIGAND_RESNAME"
[[ -n "$ANCHOR_RESIDUE" ]] && echo "Anchor:      $ANCHOR_RESIDUE"
[[ "$REFRESH" == "true" ]] && echo "Refresh:     YES"
echo ""

if [[ "$REFRESH" == "true" && -n "$RUN_NAME" ]]; then
    RUN_DIR="$CHORUS_DIR/results/$RUN_NAME"
    if [[ -d "$RUN_DIR" ]]; then
        echo "Removing existing run: $RUN_DIR"
        rm -rf "$RUN_DIR"
    fi
fi

TMPPY="$(mktemp /tmp/chorus_run_XXXXXX.py)"
trap 'rm -f "$TMPPY"' EXIT

QUIET_PY=$( [[ "$QUIET" == "true" ]] && echo "True" || echo "False" )

cat > "$TMPPY" <<PYEOF
import sys, dataclasses, types
from pathlib import Path

_repo = Path("$REPO_ROOT").resolve()
_chorus = Path("$CHORUS_DIR").resolve()
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_chorus))
if "chorus" not in sys.modules:
    _pkg = types.ModuleType("chorus")
    _pkg.__path__ = [str(_chorus)]
    sys.modules["chorus"] = _pkg

from chorus.config import Config, Paths
from chorus.experiment import run_experiment

paths = Paths.from_root(_repo)
paths = dataclasses.replace(
    paths,
    results_dir=_chorus / "results",
    data_dir=_chorus / "data",
    models_dir=Path("$MODELS_DIR"),
    checkpoints_dir=Path("$CHECKPOINTS_DIR"),
    envs_dir=Path("$ENVS_DIR"),
    cache_dir=_chorus / "cache",
)
paths.makedirs()

extra = {}
if "$GENERATORS":
    extra["generators"] = tuple(g.strip() for g in "$GENERATORS".split(","))
if "$LIGAND_RESNAME":
    extra["ligand_resname_preference"] = "$LIGAND_RESNAME"

cfg = Config(
    pdb_id="$PDB_ID",
    target_name="$TARGET_NAME",
    generator_mode="$MODE",
    scorer="$SCORER",
    max_pca_points=int("$MAX_PCA_POINTS"),
    max_umap_points=int("$MAX_UMAP_POINTS"),
    max_cluster_points=int("$MAX_CLUSTER_POINTS"),
    ecfp_family_sim_threshold=float("$ECFP_FAMILY_SIM_THRESHOLD"),
    max_tanimoto_refs_per_pocket=int("$MAX_TANIMOTO_REFS_PER_POCKET"),
    tanimoto_top_k=int("$TANIMOTO_TOP_K"),
    n_generate_per_model_per_pocket=$N_PER_POCKET,
    p2rank_max_pockets=int("$P2RANK_MAX_POCKETS"),
    p2rank_min_score=float("$P2RANK_MIN_SCORE"),
    pocket_radius_angstrom=float("$POCKET_RADIUS"),
    pocket_bbox_size_angstrom=float("$POCKET_BBOX_SIZE"),
    quiet=${QUIET_PY},
    paths=paths,
    **extra,
)

run_experiment(
    cfg,
    run_name="$RUN_NAME" if "$RUN_NAME" else None,
    anchor_residue="$ANCHOR_RESIDUE" if "$ANCHOR_RESIDUE" else None,
)
PYEOF

exec "$PYTHON" "$TMPPY"
