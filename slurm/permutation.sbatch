#!/bin/bash
# Per-subject permutation test of the baseline (after `decode baseline`).
#   sbatch --array=0-39 slurm/permutation.sbatch
#SBATCH --job-name=n400-perm
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
cd "$SLURM_SUBMIT_DIR"
python -m n400fp permutation --subject-index "$SLURM_ARRAY_TASK_ID" \
    --n-permutations "${N_PERMUTATIONS:-1000}" --n-jobs "$SLURM_CPUS_PER_TASK"
