#!/usr/bin/env bash
set -euo pipefail

CHORUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults are chosen for the six-day term-project run:
# 5 targets x 4 pockets x 3 generators x 1000 samples ~= 60k raw samples.
RUN_PREFIX="${RUN_PREFIX:-panel_$(date +%Y%m%d)}"
N_PER_POCKET="${N_PER_POCKET:-1000}"
P2RANK_MAX_POCKETS="${P2RANK_MAX_POCKETS:-4}"
P2RANK_MIN_SCORE="${P2RANK_MIN_SCORE:-0.5}"
GENERATORS="${GENERATORS:-DiffSBDD,PocketXMol,PocketXMolAR}"
MODE="${MODE:-real}"
QUIET="${QUIET:-true}"
REFRESH="${REFRESH:-false}"
DRY_RUN="${DRY_RUN:-false}"

LOG_DIR="${CHORUS_DIR}/logs/${RUN_PREFIX}"
mkdir -p "$LOG_DIR"

# Format: "PDB_ID TARGET_NAME"
# Keep this panel small enough that Boltz scoring completes with time left for analysis.
TARGETS=(
    "2PN7 GGCT"
    "4W9H VHL"
    "4CI2 CRBN"
    "3MXF BRD4_BD1"
    "5P9J BTK"
    # Optional swap-in if you want a protein-protein-interaction pocket:
    # "4HG7 MDM2"
)

slugify() {
    printf "%s" "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//; s/_$//'
}

run_target() {
    local pdb_id="$1"
    local target_name="$2"
    local target_slug
    local run_name
    local log_path

    target_slug="$(slugify "$target_name")"
    run_name="${RUN_PREFIX}_${target_slug}"
    log_path="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "=== Target ${target_name} (${pdb_id}) ==="
    echo "Run:        ${run_name}"
    echo "Generators: ${GENERATORS}"
    echo "Samples:    ${N_PER_POCKET} per generator-pocket"
    echo "Pockets:    top ${P2RANK_MAX_POCKETS} P2Rank pockets"
    echo "Log:        ${log_path}"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY_RUN=true, skipping execution."
        return 0
    fi

    (
        cd "$CHORUS_DIR"
        PDB_ID="$pdb_id" \
        TARGET_NAME="$target_name" \
        RUN_NAME="$run_name" \
        N_PER_POCKET="$N_PER_POCKET" \
        P2RANK_MAX_POCKETS="$P2RANK_MAX_POCKETS" \
        P2RANK_MIN_SCORE="$P2RANK_MIN_SCORE" \
        GENERATORS="$GENERATORS" \
        MODE="$MODE" \
        QUIET="$QUIET" \
        REFRESH="$REFRESH" \
        bash "$CHORUS_DIR/run.sh"
    ) 2>&1 | tee "$log_path"
}

echo "=== Chorus target panel ==="
echo "Run prefix: ${RUN_PREFIX}"
echo "Results:    ${CHORUS_DIR}/results"
echo "Logs:       ${LOG_DIR}"
echo ""

for target in "${TARGETS[@]}"; do
    read -r pdb_id target_name <<< "$target"
    run_target "$pdb_id" "$target_name"
done

echo ""
echo "Panel complete."
