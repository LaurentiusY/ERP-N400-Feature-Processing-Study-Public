#!/bin/bash
# Morlet coefficients for the time-frequency sensitivity analysis (guide 13.10), one subject per
# array task; needs the preprocessed data (clean-raw.fif) of the dataset in N400_DATASET.
#   sbatch --array=0-39 slurm/tfr.sbatch
#SBATCH --job-name=n400-tfr
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
cd "$SLURM_SUBMIT_DIR"
python -m n400fp tfr --subject-index "$SLURM_ARRAY_TASK_ID" --n-jobs "$SLURM_CPUS_PER_TASK"
