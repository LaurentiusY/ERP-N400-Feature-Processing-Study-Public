#!/bin/bash
# Decoding of one subject per array task for a stage (baseline | pilot | full).
# Finished feature groups are skipped, so the same command can be resubmitted.
#   sbatch --array=0-39 slurm/decode.sbatch baseline
#   sbatch --array=0-39 --time=2:00:00 slurm/decode.sbatch pilot
#   sbatch --array=0-39 --time=12:00:00 --cpus-per-task=16 slurm/decode.sbatch full
#   sbatch --array=0-39 --time=8:00:00 --cpus-per-task=16 slurm/decode.sbatch full lr extended
#     (C-grid sensitivity analysis: LR only, results in <results>/c_grid_extended/)
#   sbatch --array=0-39 --time=0:30:00 --cpus-per-task=5 slurm/decode.sbatch tfr lr
#     (time-frequency sensitivity analysis, guide 13.10: LR only, <results>/tfr_sensitivity/)
#SBATCH --job-name=n400-decode
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
STAGE="${1:?usage: sbatch --array=... slurm/decode.sbatch baseline|pilot|full [models] [c-grid]}"
MODELS="${2:-lr,lda}"
C_GRID="${3:-study}"
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
cd "$SLURM_SUBMIT_DIR"
python -m n400fp decode --stage "$STAGE" --models "$MODELS" --c-grid "$C_GRID" \
    --subject-index "$SLURM_ARRAY_TASK_ID" --n-jobs "$SLURM_CPUS_PER_TASK"
