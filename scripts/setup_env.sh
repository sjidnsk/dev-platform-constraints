#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="conda"
ENV_NAME="dev-platform-constraints"
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
    --env-name)
      ENV_NAME="$2"
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
用法：bash scripts/setup_env.sh [--conda conda] [--env-name dev-platform-constraints] [--run-validation] [--dry-run]

创建或更新 Conda 环境，以 editable 模式安装本包，并可选运行项目验证命令。
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
  "$CONDA_BIN" run -n "$ENV_NAME" python --version >/dev/null 2>&1
}

create_or_update_env() {
  if conda_env_exists; then
    "$CONDA_BIN" env update -n "$ENV_NAME" -f "$ENVIRONMENT_FILE" --prune
  else
    "$CONDA_BIN" env create -n "$ENV_NAME" -f "$ENVIRONMENT_FILE"
  fi
}

echo "Repository: $REPO_ROOT"
echo "Conda environment: $ENV_NAME"
echo "Environment file: $ENVIRONMENT_FILE"

run_step \
  "$CONDA_BIN env create/update -n \"$ENV_NAME\" -f \"$ENVIRONMENT_FILE\"" \
  create_or_update_env

run_step \
  "$CONDA_BIN install -n \"$ENV_NAME\" -c conda-forge $PYTHON_SPEC --yes" \
  "$CONDA_BIN" install -n "$ENV_NAME" -c conda-forge "$PYTHON_SPEC" --yes

run_step \
  "$CONDA_BIN run -n \"$ENV_NAME\" python -c \"$PYTHON_VERSION_CHECK\"" \
  "$CONDA_BIN" run -n "$ENV_NAME" python -c "$PYTHON_VERSION_CHECK"

run_step \
  "$CONDA_BIN run -n \"$ENV_NAME\" python -m pip install -e \"$REPO_ROOT\"" \
  "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -e "$REPO_ROOT"

if [[ "$RUN_VALIDATION" -eq 1 ]]; then
  run_step \
    "$CONDA_BIN run -n \"$ENV_NAME\" python -m unittest discover -s tests" \
    "$CONDA_BIN" run -n "$ENV_NAME" python -m unittest discover -s tests

  run_step \
    "$CONDA_BIN run -n \"$ENV_NAME\" python scripts/run_minimal_closure.py" \
    "$CONDA_BIN" run -n "$ENV_NAME" python "$REPO_ROOT/scripts/run_minimal_closure.py"
fi

cat <<NEXT

Next commands:
  conda activate $ENV_NAME
  python -m unittest discover -s tests
  python scripts/run_minimal_closure.py
NEXT
