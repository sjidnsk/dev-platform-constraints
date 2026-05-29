#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="conda"
ENV_PREFIX="${LUNAR_EXPLORER_ENV:-D:/conda_envs/lunar-explorer}"
PYTHON_SPEC="python=3.12"
PYTHON_VERSION_CHECK="import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.version)"
RUN_VALIDATION=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda)
      CONDA_BIN="$2"
      shift 2
      ;;
    --env-prefix)
      ENV_PREFIX="$2"
      shift 2
      ;;
    --run-validation)
      RUN_VALIDATION=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
用法：bash scripts/setup_env.sh [--conda conda] [--env-prefix D:/conda_envs/lunar-explorer] [--run-validation] [--dry-run]

创建或更新共享 Conda 环境。项目源码通过 PYTHONPATH 暴露，不安装 editable 本地包。
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENVIRONMENT_FILE="$REPO_ROOT/environment.yml"
SOURCE_PATH="$REPO_ROOT/src"

run_step() {
  local display="$1"
  shift
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY RUN] $display"
    return
  fi

  echo "==> $display"
  "$@"
}

conda_env_exists() {
  "$CONDA_BIN" run -p "$ENV_PREFIX" python --version >/dev/null 2>&1
}

create_or_update_env() {
  if conda_env_exists; then
    "$CONDA_BIN" env update -p "$ENV_PREFIX" -f "$ENVIRONMENT_FILE" --prune
  else
    "$CONDA_BIN" env create -p "$ENV_PREFIX" -f "$ENVIRONMENT_FILE"
  fi
}

echo "Repository: $REPO_ROOT"
echo "Conda environment: $ENV_PREFIX"
echo "Environment file: $ENVIRONMENT_FILE"
echo "Project PYTHONPATH: $SOURCE_PATH"

run_step \
  "$CONDA_BIN env create/update -p \"$ENV_PREFIX\" -f \"$ENVIRONMENT_FILE\"" \
  create_or_update_env

run_step \
  "$CONDA_BIN install -p \"$ENV_PREFIX\" -c conda-forge $PYTHON_SPEC --yes" \
  "$CONDA_BIN" install -p "$ENV_PREFIX" -c conda-forge "$PYTHON_SPEC" --yes

run_step \
  "$CONDA_BIN run -p \"$ENV_PREFIX\" python -c \"$PYTHON_VERSION_CHECK\"" \
  "$CONDA_BIN" run -p "$ENV_PREFIX" python -c "$PYTHON_VERSION_CHECK"

if [[ "$RUN_VALIDATION" -eq 1 ]]; then
  run_step \
    "PYTHONPATH=\"$SOURCE_PATH\" $CONDA_BIN run -p \"$ENV_PREFIX\" python -m unittest discover -s tests" \
    env PYTHONPATH="$SOURCE_PATH" "$CONDA_BIN" run -p "$ENV_PREFIX" python -m unittest discover -s tests

  run_step \
    "PYTHONPATH=\"$SOURCE_PATH\" $CONDA_BIN run -p \"$ENV_PREFIX\" python scripts/run_minimal_closure.py" \
    env PYTHONPATH="$SOURCE_PATH" "$CONDA_BIN" run -p "$ENV_PREFIX" python "$REPO_ROOT/scripts/run_minimal_closure.py"
fi

cat <<NEXT

Next commands:
  conda activate $ENV_PREFIX
  export PYTHONPATH=src
  python -m unittest discover -s tests
  python scripts/run_minimal_closure.py
NEXT
