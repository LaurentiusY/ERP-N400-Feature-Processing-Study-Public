#!/bin/bash
# EEGNet generality check, one subject per array task. FIR requires a GPU type; the smallest
# H100 slice (1g.10gb) is ample for EEGNet. For a full GPU: sbatch --gpus=h100:1 ...
#   sbatch --array=0-2  slurm/eegnet.sbatch                                   # gate 1, baseline
#   sbatch --array=0-39 slurm/eegnet.sbatch "$N400_RESULTS_DIR/reports/full/eegnet_configs.txt"
#SBATCH --job-name=n400-eegnet
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
cd "$SLURM_SUBMIT_DIR"
EXTRA=()
if [ -n "${1:-}" ]; then EXTRA=(--config-file "$1"); fi
python -m n400fp eegnet --subject-index "$SLURM_ARRAY_TASK_ID" --device cuda "${EXTRA[@]}"
