#!/usr/bin/env bash
set -euo pipefail

CHORUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULTS_GLOB="${RESULTS_GLOB:-results/*}"
DEFAULT_SCORER="${DEFAULT_SCORER:-rtmscore}"
SCORER_OVERRIDE="${SCORER:-}"
QUIET="${QUIET:-true}"
DRY_RUN="${DRY_RUN:-false}"
JOBS="${JOBS:-auto}"
AUTO_MAX_JOBS="${AUTO_MAX_JOBS:-4}"
MEM_GB_PER_JOB="${MEM_GB_PER_JOB:-8}"
PYTHON_JSON="${PYTHON_JSON:-python3}"
LOG_DIR="${LOG_DIR:-$CHORUS_DIR/logs/rebuild_figures_$(date +%Y%m%d_%H%M%S)}"

usage() {
    cat <<EOF
Usage: bash rebuild_figures.sh [RESULT_DIR ...]

Rebuild analysis CSVs and figures for completed cached runs.

Examples:
  bash rebuild_figures.sh
  DRY_RUN=true bash rebuild_figures.sh
  RESULTS_GLOB='results/panel_20260430_*' bash rebuild_figures.sh

Environment:
  RESULTS_GLOB       Result directories to scan when no positional args are given.
  SCORER             Override scorer cache to load, e.g. rtmscore.
  DEFAULT_SCORER     Used when run_summary.json has no scorer. Default: rtmscore.
  QUIET              Passed through to run.sh. Default: true.
  DRY_RUN            Print planned rebuilds without running them. Default: false.
  JOBS               Number of target runs to rebuild in parallel, or auto. Default: auto.
  AUTO_MAX_JOBS      Upper bound when JOBS=auto. Default: 4.
  MEM_GB_PER_JOB     Memory budget per parallel rebuild when JOBS=auto. Default: 8.
  LOG_DIR            Where per-target rebuild logs are written.
  MAX_PCA_POINTS     Passed through to run.sh if set.
  MAX_CLUSTER_POINTS Passed through to run.sh if set.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

scored_cache_name() {
    local scorer="$1"
    echo "scored_candidates_${scorer}.csv"
}

detect_cpus() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    else
        getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
    fi
}

detect_mem_gib() {
    awk '/MemAvailable:/ {print int($2 / 1024 / 1024)}' /proc/meminfo 2>/dev/null || echo 0
}

min_int() {
    local min="$1"
    shift
    local value
    for value in "$@"; do
        if ((value < min)); then
            min="$value"
        fi
    done
    echo "$min"
}

resolve_jobs() {
    local requested="$1"
    local n_runs="$2"
    local cpus mem_gib cpu_jobs mem_jobs resolved

    if [[ "$requested" != "auto" ]]; then
        if ! [[ "$requested" =~ ^[0-9]+$ ]] || ((requested < 1)); then
            echo "Invalid JOBS=$requested; expected auto or a positive integer." >&2
            exit 1
        fi
        echo "$requested"
        return
    fi

    cpus="$(detect_cpus)"
    mem_gib="$(detect_mem_gib)"
    cpu_jobs=$((cpus / 4))
    ((cpu_jobs < 1)) && cpu_jobs=1

    if ((mem_gib > 0)); then
        mem_jobs=$((mem_gib / MEM_GB_PER_JOB))
        ((mem_jobs < 1)) && mem_jobs=1
    else
        mem_jobs="$AUTO_MAX_JOBS"
    fi

    resolved="$(min_int "$n_runs" "$AUTO_MAX_JOBS" "$cpu_jobs" "$mem_jobs")"
    ((resolved < 1)) && resolved=1
    echo "$resolved"
}

read_summary() {
    local summary_path="$1"
    "$PYTHON_JSON" - "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())

def clean(value):
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\n", " ").strip()

run_name = clean(data.get("run_name")) or path.parent.name
pdb_id = clean(data.get("pdb_id"))
target_name = clean(data.get("target_name")) or pdb_id or run_name
scorer = clean(data.get("scorer"))
print("\t".join([run_name, pdb_id, target_name, scorer]))
PY
}

run_dirs=()
if (($# > 0)); then
    run_dirs=("$@")
else
    shopt -s nullglob
    # shellcheck disable=SC2206
    run_dirs=($RESULTS_GLOB)
    shopt -u nullglob
fi

if ((${#run_dirs[@]} == 0)); then
    echo "No result directories matched."
    echo "Set RESULTS_GLOB or pass result directories explicitly."
    echo "Nothing to rebuild."
    exit 0
fi

RESOLVED_JOBS="$(resolve_jobs "$JOBS" "${#run_dirs[@]}")"
CPUS_TOTAL="$(detect_cpus)"
THREADS_PER_JOB=$((CPUS_TOTAL / RESOLVED_JOBS))
((THREADS_PER_JOB < 1)) && THREADS_PER_JOB=1

if [[ "$DRY_RUN" != "true" ]]; then
    mkdir -p "$LOG_DIR"
fi

echo "=== Chorus figure rebuild ==="
echo "Results matched: ${#run_dirs[@]}"
echo "Scorer override: ${SCORER_OVERRIDE:-<from run_summary.json>}"
echo "Default scorer:  ${DEFAULT_SCORER}"
echo "Dry run:         ${DRY_RUN}"
echo "Parallel jobs:   ${RESOLVED_JOBS} (${JOBS})"
echo "Threads/job:     ${THREADS_PER_JOB}"
[[ "$DRY_RUN" != "true" ]] && echo "Logs:            ${LOG_DIR}"
echo ""

rebuilt=0
skipped=0
failed=0
candidate_count=0
active_pids=()
active_names=()
failed_names=()

run_rebuild() {
    local run_name="$1"
    local pdb_id="$2"
    local target_name="$3"
    local scorer="$4"
    local log_path="$5"

    (
        cd "$CHORUS_DIR"
        PDB_ID="$pdb_id" \
        TARGET_NAME="$target_name" \
        RUN_NAME="$run_name" \
        SCORER="$scorer" \
        QUIET="$QUIET" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS_PER_JOB}" \
        MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS_PER_JOB}" \
        OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$THREADS_PER_JOB}" \
        NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-$THREADS_PER_JOB}" \
        REFRESH=false \
        bash "$CHORUS_DIR/run.sh"
    ) > "$log_path" 2>&1
}

wait_for_oldest() {
    local pid="${active_pids[0]}"
    local name="${active_names[0]}"
    if wait "$pid"; then
        echo "DONE:  $name"
        rebuilt=$((rebuilt + 1))
    else
        echo "FAIL:  $name (see $LOG_DIR/${name}.log)"
        failed=$((failed + 1))
        failed_names+=("$name")
    fi
    active_pids=("${active_pids[@]:1}")
    active_names=("${active_names[@]:1}")
}

for run_dir in "${run_dirs[@]}"; do
    run_dir="${run_dir%/}"
    summary_path="$run_dir/run_summary.json"

    if [[ ! -d "$run_dir" ]]; then
        echo "SKIP: $run_dir is not a directory."
        skipped=$((skipped + 1))
        continue
    fi
    if [[ ! -f "$summary_path" ]]; then
        echo "SKIP: $run_dir has no run_summary.json."
        skipped=$((skipped + 1))
        continue
    fi

    IFS=$'\t' read -r run_name pdb_id target_name summary_scorer < <(read_summary "$summary_path")
    scorer="${SCORER_OVERRIDE:-${summary_scorer:-$DEFAULT_SCORER}}"
    scored_cache="$(scored_cache_name "$scorer")"

    missing=()
    [[ -f "$run_dir/generated_by_generator.csv" ]] || missing+=("generated_by_generator.csv")
    [[ -f "$run_dir/unique_generated.csv" ]] || missing+=("unique_generated.csv")
    [[ -f "$run_dir/$scored_cache" ]] || missing+=("$scored_cache")

    if ((${#missing[@]} > 0)); then
        echo "SKIP: $run_dir is missing cached input(s): ${missing[*]}"
        skipped=$((skipped + 1))
        continue
    fi

    echo "REBUILD: $run_name ($target_name, $pdb_id) using $scored_cache"
    candidate_count=$((candidate_count + 1))
    if [[ "$DRY_RUN" == "true" ]]; then
        continue
    fi

    log_path="$LOG_DIR/${run_name}.log"
    echo "START: $run_name -> $log_path"
    run_rebuild "$run_name" "$pdb_id" "$target_name" "$scorer" "$log_path" &
    active_pids+=("$!")
    active_names+=("$run_name")

    if ((${#active_pids[@]} >= RESOLVED_JOBS)); then
        wait_for_oldest
    fi
done

while ((${#active_pids[@]} > 0)); do
    wait_for_oldest
done

echo ""
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run complete. Rebuild candidates: ${candidate_count}; skipped: ${skipped}."
else
    echo "Figure rebuild complete. Rebuilt: ${rebuilt}; skipped: ${skipped}; failed: ${failed}."
    if ((${failed} > 0)); then
        echo "Failed runs: ${failed_names[*]}"
        exit 1
    fi
fi
