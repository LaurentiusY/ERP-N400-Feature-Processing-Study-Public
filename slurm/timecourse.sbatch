#!/bin/bash
# Time-resolved decoding curves (section 6.5), one subject per array task.
#   sbatch --array=0-39 slurm/timecourse.sbatch
#SBATCH --job-name=n400-timecourse
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
cd "$SLURM_SUBMIT_DIR"
python -m n400fp timecourse --subject-index "$SLURM_ARRAY_TASK_ID"
