#!/bin/bash
# Download the ERP CORE N400 recordings (40 subjects, ~2.5 GB) from OSF into $N400_DATA_DIR
# with hu-neuro-pipeline (FIR compute nodes have internet). Files are md5-checked and
# existing ones are skipped, so a failed download can be resubmitted unchanged.
#   sbatch slurm/download.sbatch
#SBATCH --job-name=n400-download
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
cd "$SLURM_SUBMIT_DIR"
python -m n400fp download
echo "EEGLAB files: $(find "$N400_DATA_DIR" -name '*N400_eeg.set' | wc -l) .set," \
    "$(find "$N400_DATA_DIR" -name '*N400_eeg.fdt' | wc -l) .fdt (expected 40 each)"
du -sh "$N400_DATA_DIR"
