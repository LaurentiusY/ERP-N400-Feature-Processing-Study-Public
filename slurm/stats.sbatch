#!/bin/bash
# Group statistics of a stage; the exploratory pairwise comparisons need R (afex, emmeans).
#   sbatch slurm/stats.sbatch baseline|pilot|full
#SBATCH --job-name=n400-stats
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
STAGE="${1:?usage: sbatch slurm/stats.sbatch baseline|pilot|full}"
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
cd "$SLURM_SUBMIT_DIR"
python -m n400fp stats --stage "$STAGE"
if [ "$STAGE" = full ]; then
    bash slurm/run_pairwise.sh || echo "pairwise comparisons skipped; later: sbatch slurm/pairwise.sbatch"
fi
