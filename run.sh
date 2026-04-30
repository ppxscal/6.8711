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

_find_python() {
    for candidate in \
        "$ENV_ROOT/chorus/bin/python" \
        "$ENV_ROOT/diffsbdd/bin/python" \
        "$REPO_ROOT/unicellular/.venv/bin/python" \
        "$(which python3 2>/dev/null)"; do
        if [[ -x "$candidate" ]] && "$candidate" -c "import hdbscan, rdkit, pandas, torch" 2>/dev/null; then
            echo "$candidate"
            return 0
        fi
    done
    echo "Error: could not find a Python with hdbscan, rdkit, pandas, and torch installed." >&2
    echo "Run: bash setup.sh diffsbdd" >&2
    return 1
}
PYTHON="$(_find_python)"

if [[ -z "${PDB_ID:-}" ]]; then
    echo "Error: PDB_ID is required."
    echo "Usage: PDB_ID=2PN7 TARGET_NAME=GGCT bash run.sh"
    exit 1
fi

TARGET_NAME="${TARGET_NAME:-$PDB_ID}"
N_PER_POCKET="${N_PER_POCKET:-1250}"   # 2 gens × 4 pockets × 1250 = 10k total
MODE="${MODE:-real}"
RUN_NAME="${RUN_NAME:-}"
QUIET="${QUIET:-true}"
REFRESH="${REFRESH:-false}"
GENERATORS="${GENERATORS:-}"
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

from chorus.pipeline import Config, Paths, run_pipeline

paths = Paths.from_root(_repo)
paths = dataclasses.replace(
    paths,
    results_dir=_chorus / "results",
    data_dir=_chorus / "data",
    models_dir=Path("$MODELS_DIR"),
    checkpoints_dir=Path("$CHECKPOINTS_DIR"),
    envs_dir=Path("$ENVS_DIR"),
    cache_dir=_chorus / "cache",
    boltz_cache_dir=_chorus / "cache" / "boltz",
    msa_cache_dir=_chorus / "cache" / "boltz" / "msa",
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
    n_generate_per_model_per_pocket=$N_PER_POCKET,
    p2rank_max_pockets=int("$P2RANK_MAX_POCKETS"),
    p2rank_min_score=float("$P2RANK_MIN_SCORE"),
    pocket_radius_angstrom=float("$POCKET_RADIUS"),
    pocket_bbox_size_angstrom=float("$POCKET_BBOX_SIZE"),
    quiet=${QUIET_PY},
    paths=paths,
    **extra,
)

run_pipeline(
    cfg,
    run_name="$RUN_NAME" if "$RUN_NAME" else None,
    anchor_residue="$ANCHOR_RESIDUE" if "$ANCHOR_RESIDUE" else None,
)
PYEOF

exec "$PYTHON" "$TMPPY"
