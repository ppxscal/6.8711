#!/usr/bin/env bash
set -euo pipefail

CHORUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$CHORUS_DIR")"

UV="${UV:-$(command -v uv || true)}"
ENV_ROOT="${ENV_ROOT:-$REPO_ROOT/envs/uv}"
PY311="${PY311:-/usr/bin/python3.11}"
PY310="${PY310:-/usr/bin/python3.10}"
FORCE="${FORCE:-false}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$CHORUS_DIR/cache/numba}"
mkdir -p "$NUMBA_CACHE_DIR"

# Package specs are centralized here so a presentation/demo can state exactly
# what was installed. Override any of these env vars to test a newer package.
CHORUS_PACKAGES=(
    "numpy>=1.26,<2"
    "pandas>=2.2"
    "matplotlib"
    "rdkit>=2024.3.2"
    "scikit-learn>=1.4"
    "tqdm"
    "hdbscan"
    "seaborn"
)
BOLTZ_SPEC="${BOLTZ_SPEC:-boltz==2.2.1}"

usage() {
    cat <<EOF
Usage: bash setup.sh [target]

Targets:
  chorus          Build the lightweight orchestration/analysis env.
  boltz           Build the Boltz scoring env.
  check           Check envs and expected console scripts.
  all             Build chorus + boltz, then check.

Environment overrides:
  FORCE=true      Recreate the selected env from scratch.
  ENV_ROOT=PATH   Default: $ENV_ROOT
  PY311=PATH      Default: $PY311
  BOLTZ_SPEC=SPEC Default: $BOLTZ_SPEC

Examples:
  bash setup.sh all
  FORCE=true bash setup.sh boltz
  BOLTZ_SPEC='boltz[cuda]==2.2.1' FORCE=true bash setup.sh boltz
EOF
}

require_uv() {
    if [[ -z "$UV" ]]; then
        echo "Error: uv is not on PATH. Install uv or set UV=/path/to/uv." >&2
        exit 1
    fi
}

require_python() {
    local py="$1"
    if [[ ! -x "$py" ]]; then
        echo "Error: required Python is not executable: $py" >&2
        exit 1
    fi
}

env_python() {
    local name="$1"
    printf "%s/bin/python" "$ENV_ROOT/$name"
}

env_script() {
    local name="$1"
    local script="$2"
    printf "%s/bin/%s" "$ENV_ROOT/$name" "$script"
}

python_works() {
    local py="$1"
    [[ -x "$py" ]] && "$py" -c "import sys; print(sys.executable)" >/dev/null 2>&1
}

make_env() {
    local name="$1"
    local py="$2"
    local env_dir="$ENV_ROOT/$name"
    local env_py
    env_py="$(env_python "$name")"

    require_uv
    require_python "$py"
    mkdir -p "$ENV_ROOT"

    if [[ "$FORCE" == "true" && -e "$env_dir" ]]; then
        echo "Recreating $env_dir"
        rm -rf "$env_dir"
    fi

    if python_works "$env_py"; then
        echo "Using existing env: $env_dir"
        return 0
    fi

    if [[ -e "$env_dir" ]]; then
        echo "Env exists but its Python is not usable: $env_dir" >&2
        echo "Rerun with FORCE=true bash setup.sh $name" >&2
        exit 1
    fi

    echo "Creating env: $env_dir with $py"
    "$UV" venv --python "$py" "$env_dir"
}

uv_install() {
    local name="$1"
    shift
    UV_LINK_MODE=copy "$UV" pip install --python "$(env_python "$name")" "$@"
}

setup_chorus() {
    make_env chorus "$PY311"
    uv_install chorus --upgrade pip
    uv_install chorus --upgrade "${CHORUS_PACKAGES[@]}"
    "$(env_python chorus)" - <<'PY'
import hdbscan
import matplotlib
import numpy
import pandas
import rdkit
import sklearn
print("chorus env ok")
PY
}

setup_boltz() {
    make_env boltz "$PY311"
    uv_install boltz --upgrade pip
    uv_install boltz --upgrade "$BOLTZ_SPEC"
    NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR" "$(env_python boltz)" "$(env_script boltz boltz)" --help >/dev/null
    echo "boltz env ok: $(env_script boltz boltz)"
}

check_env() {
    local name="$1"
    local py
    py="$(env_python "$name")"
    if python_works "$py"; then
        echo "OK  $name python: $py"
    else
        echo "BAD $name python: $py"
        return 1
    fi
}

check_script() {
    local name="$1"
    local script="$2"
    local path
    path="$(env_script "$name" "$script")"
    if [[ -x "$path" ]]; then
        echo "OK  $name script: $path"
    else
        echo "BAD $name script: $path"
        return 1
    fi
}

check_all() {
    local status=0
    check_env chorus || status=1
    check_env boltz || status=1
    check_script boltz boltz || status=1
    check_env diffsbdd || status=1
    check_env pocketxmol || status=1

    if [[ "$status" -eq 0 ]]; then
        echo "All checked envs are present."
    else
        echo "One or more env checks failed." >&2
    fi
    return "$status"
}

target="${1:-all}"
case "$target" in
    chorus)
        setup_chorus
        ;;
    boltz)
        setup_boltz
        ;;
    check)
        check_all
        ;;
    all)
        setup_chorus
        setup_boltz
        check_all
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown setup target: $target" >&2
        usage >&2
        exit 1
        ;;
esac
