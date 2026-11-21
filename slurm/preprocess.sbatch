#!/bin/bash
# Sections 4 and 5 for one subject per array task.
#   sbatch --array=0-39 slurm/preprocess.sbatch          # all subjects
#   sbatch --array=0-2  slurm/preprocess.sbatch          # gate 1: three subjects
#SBATCH --job-name=n400-preprocess
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
cd "$SLURM_SUBMIT_DIR"
python -m n400fp preprocess --subject-index "$SLURM_ARRAY_TASK_ID" --n-jobs "$SLURM_CPUS_PER_TASK"
python -m n400fp bands --subject-index "$SLURM_ARRAY_TASK_ID" --n-jobs "$SLURM_CPUS_PER_TASK"
