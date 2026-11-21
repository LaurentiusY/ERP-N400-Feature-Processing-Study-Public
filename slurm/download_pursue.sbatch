#!/bin/bash
# Download the PURSUE N400 dataset (NEMAR on007052 v1.0.0, CC0; 288 participants, ~9.6 GB) into
# $N400_PURSUE_DIR, checking every file against the NEMAR manifest (sha256 for the EEG files).
# Resumable: resubmit after a failure or a time-out; verified files are skipped.
# Guide section 14 (preregistration tagged prereg-pursue-v1 before this download).
#   sbatch slurm/download_pursue.sbatch [workers]
#SBATCH --job-name=n400-pursue-dl
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
WORKERS="${1:-8}"
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python "$SLURM_SUBMIT_DIR/scripts/fetch_manifest.py" \
    "https://data.nemar.org/on007052/v1.0.0/manifest.json" "$N400_PURSUE_DIR" --workers "$WORKERS"
du -sh "$N400_PURSUE_DIR"
find "$N400_PURSUE_DIR" -name "*_eeg.set" | wc -l
