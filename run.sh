#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --job-name=satisfaction-family
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Use the existing project environment; do not create or replace it.
PYTHON_BIN="$AIRLINE_PYTHON"
if [[ -z "$PYTHON_BIN" ]]; then PYTHON_BIN="../.venv/bin/python"; fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Set AIRLINE_PYTHON to the existing project virtualenv's python." >&2
    exit 1
fi
action="predict"
if [[ $# -gt 0 ]]; then action="$1"; shift; fi
case "$action" in
    prepare) "$PYTHON_BIN" -u prepare_inputs.py "$@" ;;
    cv|train|predict) "$PYTHON_BIN" -u family_stack.py "$action" "$@" ;;
    *) echo "Usage: sbatch run.sh [prepare|cv|train|predict] [arguments]" >&2; exit 2 ;;
esac
