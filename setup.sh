#!/usr/bin/env bash
set -euo pipefail

CHORUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${CHORUS_ROOT:-$CHORUS_DIR}"

UV="${UV:-$(command -v uv || true)}"
ENV_ROOT="${ENV_ROOT:-$REPO_ROOT/envs/uv}"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$REPO_ROOT/checkpoints}"
PY311="${PY311:-/usr/bin/python3.11}"
PY310="${PY310:-/usr/bin/python3.10}"
FORCE="${FORCE:-false}"
REFRESH_DOWNLOADS="${REFRESH_DOWNLOADS:-false}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$CHORUS_DIR/cache/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CHORUS_DIR/cache/matplotlib}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

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
DIFFSBDD_REPO_URL="${DIFFSBDD_REPO_URL:-https://github.com/arneschneuing/DiffSBDD.git}"
POCKETXMOL_REPO_URL="${POCKETXMOL_REPO_URL:-https://github.com/pengxingang/PocketXMol.git}"
DIFFSBDD_CKPT_URL="${DIFFSBDD_CKPT_URL:-https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt?download=1}"
POCKETXMOL_WEIGHTS_URL="${POCKETXMOL_WEIGHTS_URL:-https://zenodo.org/records/17801271/files/model_weights.tar.gz?download=1}"
TORCH_CUDA="${TORCH_CUDA:-cu118}"
TORCH_VERSION="${TORCH_VERSION:-2.0.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.15.2}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/$TORCH_CUDA}"
PYG_FIND_LINKS="${PYG_FIND_LINKS:-https://data.pyg.org/whl/torch-${TORCH_VERSION}+${TORCH_CUDA}.html}"
TORCH_MAJOR_MINOR="${TORCH_VERSION%.*}"
PYG_TORCH_TAG="${PYG_TORCH_TAG:-pt${TORCH_MAJOR_MINOR//./}${TORCH_CUDA}}"
TORCH_SCATTER_VERSION="${TORCH_SCATTER_VERSION:-2.1.2}"
TORCH_SPARSE_VERSION="${TORCH_SPARSE_VERSION:-0.6.18}"
TORCH_CLUSTER_VERSION="${TORCH_CLUSTER_VERSION:-1.6.3}"
PYG_EXTENSION_PACKAGES=(
    "torch-scatter==${TORCH_SCATTER_VERSION}+${PYG_TORCH_TAG}"
    "torch-sparse==${TORCH_SPARSE_VERSION}+${PYG_TORCH_TAG}"
    "torch-cluster==${TORCH_CLUSTER_VERSION}+${PYG_TORCH_TAG}"
)
DIFFSBDD_PACKAGES=(
    "numpy>=1.24,<2"
    "scipy>=1.10"
    "pandas>=1.5"
    "rdkit>=2023.9.3"
    "biopython==1.79"
    "imageio"
    "pytorch-lightning==1.8.4"
    "torchmetrics==1.4.2"
    "wandb==0.13.1"
    "seaborn"
    "tqdm"
    "openbabel-wheel"
    "gemmi"
    "matplotlib"
    "tensorboardX"
    "setuptools<70"
)
POCKETXMOL_PACKAGES=(
    "numpy>=1.24,<2"
    "pandas>=1.5"
    "scipy>=1.10"
    "rdkit>=2023.9.3"
    "biopython>=1.83"
    "PeptideBuilder==1.1.0"
    "lmdb"
    "easydict"
    "pytorch-lightning==2.0.9"
    "tqdm"
    "openbabel-wheel"
    "scikit-learn"
    "requests"
    "psutil"
    "Jinja2"
    "pyparsing"
    "tensorboard"
    "setuptools<70"
)

usage() {
    cat <<EOF
Usage: bash setup.sh [target]

Targets:
  chorus          Build the lightweight orchestration/analysis env.
  boltz           Build the Boltz scoring env.
  models          Clone generator repos and download required checkpoints.
  diffsbdd        Build/download DiffSBDD repo, checkpoint, and env.
  pocketxmol      Build/download PocketXMol repo, weights, and env.
  generators      Build/download both generator stacks.
  check           Check envs and expected console scripts.
  all             Build chorus + boltz + generators, then check.

Environment overrides:
  FORCE=true      Recreate the selected env from scratch.
  REFRESH_DOWNLOADS=true Re-download model/checkpoint archives.
  CHORUS_ROOT=PATH Default: $REPO_ROOT
  ENV_ROOT=PATH   Default: $ENV_ROOT
  MODELS_DIR=PATH Default: $MODELS_DIR
  CHECKPOINTS_DIR=PATH Default: $CHECKPOINTS_DIR
  PY311=PATH      Default: $PY311
  PY310=PATH      Default: $PY310
  TORCH_CUDA=TAG  Default: $TORCH_CUDA
  PYG_TORCH_TAG=TAG Default: $PYG_TORCH_TAG
  BOLTZ_SPEC=SPEC Default: $BOLTZ_SPEC

Examples:
  bash setup.sh all
  bash setup.sh generators
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

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command is not on PATH: $cmd" >&2
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

uv_install_torch() {
    local name="$1"
    shift
    UV_LINK_MODE=copy "$UV" pip install \
        --python "$(env_python "$name")" \
        --index-url "$TORCH_INDEX_URL" \
        "$@"
}

uv_install_with_torch_index() {
    local name="$1"
    shift
    UV_LINK_MODE=copy "$UV" pip install \
        --python "$(env_python "$name")" \
        --index-url https://pypi.org/simple \
        --extra-index-url "$TORCH_INDEX_URL" \
        "$@"
}

uv_install_pyg_extensions() {
    local name="$1"
    shift
    UV_LINK_MODE=copy "$UV" pip install \
        --python "$(env_python "$name")" \
        --no-index \
        --find-links "$PYG_FIND_LINKS" \
        --no-deps \
        "$@"
}

download_file() {
    local url="$1"
    local out="$2"
    mkdir -p "$(dirname "$out")"
    if [[ -s "$out" && "$REFRESH_DOWNLOADS" != "true" ]]; then
        echo "Using existing file: $out"
        return 0
    fi
    local tmp="${out}.tmp"
    rm -f "$tmp"
    echo "Downloading $url"
    echo "        -> $out"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 --output "$tmp" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$tmp" "$url"
    else
        echo "Error: need curl or wget to download $url" >&2
        exit 1
    fi
    mv "$tmp" "$out"
}

clone_repo() {
    local url="$1"
    local dir="$2"
    require_cmd git
    mkdir -p "$(dirname "$dir")"
    if [[ -d "$dir/.git" ]]; then
        echo "Using existing repo: $dir"
        return 0
    fi
    if [[ -e "$dir" ]]; then
        echo "Error: $dir exists but is not a git repo." >&2
        echo "Move it aside or set MODELS_DIR to a clean location." >&2
        exit 1
    fi
    echo "Cloning $url"
    git clone "$url" "$dir"
}

ensure_diffsbdd_repo() {
    clone_repo "$DIFFSBDD_REPO_URL" "$MODELS_DIR/diffsbdd"
}

ensure_pocketxmol_repo() {
    clone_repo "$POCKETXMOL_REPO_URL" "$MODELS_DIR/pocketxmol"
}

ensure_diffsbdd_checkpoint() {
    download_file \
        "$DIFFSBDD_CKPT_URL" \
        "$CHECKPOINTS_DIR/diffsbdd/crossdocked_fullatom_cond.ckpt"
}

ensure_pocketxmol_weights() {
    local archive="$CHECKPOINTS_DIR/pocketxmol/model_weights.tar.gz"
    local ckpt="$CHECKPOINTS_DIR/pocketxmol/data/trained_models/pxm/checkpoints/pocketxmol.ckpt"
    local train_cfg="$CHECKPOINTS_DIR/pocketxmol/data/trained_models/pxm/train_config/train.yml"
    download_file "$POCKETXMOL_WEIGHTS_URL" "$archive"
    if [[ ! -s "$ckpt" || ! -s "$train_cfg" || "$REFRESH_DOWNLOADS" == "true" ]]; then
        echo "Extracting PocketXMol weights into $CHECKPOINTS_DIR/pocketxmol"
        tar -xzf "$archive" -C "$CHECKPOINTS_DIR/pocketxmol"
    fi
    if [[ ! -s "$ckpt" ]]; then
        echo "Error: PocketXMol checkpoint was not found after extraction: $ckpt" >&2
        exit 1
    fi
    if [[ ! -s "$train_cfg" ]]; then
        echo "Error: PocketXMol train config was not found after extraction: $train_cfg" >&2
        exit 1
    fi
}

link_pocketxmol_weights_into_repo() {
    local repo="$MODELS_DIR/pocketxmol"
    local ckpt_src="$CHECKPOINTS_DIR/pocketxmol/data/trained_models/pxm/checkpoints/pocketxmol.ckpt"
    local cfg_src="$CHECKPOINTS_DIR/pocketxmol/data/trained_models/pxm/train_config/train.yml"
    local ckpt_dst="$repo/data/trained_models/pxm/checkpoints/pocketxmol.ckpt"
    local cfg_dst="$repo/data/trained_models/pxm/train_config/train.yml"

    mkdir -p "$(dirname "$ckpt_dst")" "$(dirname "$cfg_dst")"
    if [[ -L "$ckpt_dst" ]]; then
        ln -sfn "$ckpt_src" "$ckpt_dst"
    elif [[ ! -e "$ckpt_dst" ]]; then
        ln -s "$ckpt_src" "$ckpt_dst"
    fi
    if [[ -L "$cfg_dst" ]]; then
        ln -sfn "$cfg_src" "$cfg_dst"
    elif [[ ! -e "$cfg_dst" ]]; then
        ln -s "$cfg_src" "$cfg_dst"
    fi
}

setup_models() {
    mkdir -p "$MODELS_DIR" "$CHECKPOINTS_DIR"
    ensure_diffsbdd_repo
    ensure_pocketxmol_repo
    ensure_diffsbdd_checkpoint
    ensure_pocketxmol_weights
    link_pocketxmol_weights_into_repo
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

setup_diffsbdd() {
    ensure_diffsbdd_repo
    ensure_diffsbdd_checkpoint
    make_env diffsbdd "$PY310"
    uv_install_torch diffsbdd \
        "torch==${TORCH_VERSION}+${TORCH_CUDA}" \
        "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA}"
    uv_install_with_torch_index diffsbdd --upgrade \
        "${DIFFSBDD_PACKAGES[@]}" \
        "torch==${TORCH_VERSION}+${TORCH_CUDA}" \
        "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA}"
    uv_install diffsbdd "torch-geometric==2.7.0"
    uv_install_pyg_extensions diffsbdd "${PYG_EXTENSION_PACKAGES[@]}"
    "$(env_python diffsbdd)" - <<'PY'
import Bio
import openbabel
import pytorch_lightning
import rdkit
import torch
import torch_cluster
import torch_scatter
import torch_sparse
print("diffsbdd env ok", torch.__version__)
PY
}

setup_pocketxmol() {
    ensure_pocketxmol_repo
    ensure_pocketxmol_weights
    link_pocketxmol_weights_into_repo
    make_env pocketxmol "$PY310"
    uv_install_torch pocketxmol "torch==${TORCH_VERSION}+${TORCH_CUDA}"
    uv_install_with_torch_index pocketxmol --upgrade \
        "${POCKETXMOL_PACKAGES[@]}" \
        "torch==${TORCH_VERSION}+${TORCH_CUDA}"
    uv_install pocketxmol "torch-geometric==2.3.0"
    uv_install_pyg_extensions pocketxmol "${PYG_EXTENSION_PACKAGES[@]}"
    "$(env_python pocketxmol)" - <<'PY'
import Bio
import easydict
import lmdb
import openbabel
import rdkit
import torch
import torch_cluster
import torch_geometric
import torch_scatter
import torch_sparse
print("pocketxmol env ok", torch.__version__)
PY
}

setup_generators() {
    setup_diffsbdd
    setup_pocketxmol
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

check_imports() {
    local name="$1"
    local code="$2"
    local py
    py="$(env_python "$name")"
    if [[ -x "$py" ]] && "$py" -c "$code" >/dev/null 2>&1; then
        echo "OK  $name imports"
    else
        echo "BAD $name imports"
        return 1
    fi
}

check_path() {
    local label="$1"
    local path="$2"
    if [[ -e "$path" ]]; then
        echo "OK  $label: $path"
    else
        echo "BAD $label: $path"
        return 1
    fi
}

check_all() {
    local status=0
    check_env chorus || status=1
    check_imports chorus "import hdbscan, matplotlib, numpy, pandas, rdkit, sklearn" || status=1
    check_env boltz || status=1
    check_script boltz boltz || status=1
    check_env diffsbdd || status=1
    check_imports diffsbdd "from openbabel import openbabel; from Bio.PDB.Polypeptide import three_to_one; import pytorch_lightning, rdkit, torch, torch_scatter, torch_sparse, torch_cluster" || status=1
    check_env pocketxmol || status=1
    check_imports pocketxmol "from openbabel import openbabel; import Bio, easydict, lmdb, rdkit, torch, torch_geometric, torch_scatter, torch_sparse, torch_cluster" || status=1
    check_path "diffsbdd repo" "$MODELS_DIR/diffsbdd/generate_ligands.py" || status=1
    check_path "diffsbdd checkpoint" "$CHECKPOINTS_DIR/diffsbdd/crossdocked_fullatom_cond.ckpt" || status=1
    check_path "pocketxmol repo" "$MODELS_DIR/pocketxmol/scripts/sample_use.py" || status=1
    check_path "pocketxmol checkpoint" "$CHECKPOINTS_DIR/pocketxmol/data/trained_models/pxm/checkpoints/pocketxmol.ckpt" || status=1
    check_path "pocketxmol repo checkpoint link" "$MODELS_DIR/pocketxmol/data/trained_models/pxm/checkpoints/pocketxmol.ckpt" || status=1

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
    models|assets)
        setup_models
        ;;
    diffsbdd)
        setup_diffsbdd
        ;;
    pocketxmol)
        setup_pocketxmol
        ;;
    generators)
        setup_generators
        ;;
    check)
        check_all
        ;;
    all)
        setup_chorus
        setup_boltz
        setup_generators
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
