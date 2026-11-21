#!/bin/bash
# Download the Draschkow et al. (2018) dataset (Zenodo 10.5281/zenodo.1421347, CC BY 4.0;
# visual-scene and auditory-sentence N400, 40 participants; one 26.8 GB zip), check its MD5
# and unzip it into $N400_ROOT/external/draschkow2018.
# The zip was written on macOS without zip64 records (offsets truncated to 32 bits), so it is
# extracted with scripts/unzip_truncated_zip.py (CRC-checked) instead of unzip.
# Zenodo limits each connection (a single connection from FIR ran at ~45 kB/s), so the file is
# fetched with parallel range requests (scripts/fetch_parallel.py). Resumable: resubmit after
# a failure or a time-out; finished parts are kept. Default 8 connections (Zenodo blocks bursts).
#   sbatch slurm/download_draschkow.sbatch [workers]
#SBATCH --job-name=n400-zenodo
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
WORKERS="${1:-8}"
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
dest="$N400_ROOT/external/draschkow2018"
mkdir -p "$dest"
python "$SLURM_SUBMIT_DIR/scripts/fetch_parallel.py" \
    "https://zenodo.org/api/records/1421347/files/n400Data.zip/content" \
    "$dest/n400Data.zip" --size 26830252725 --md5 4dca2ee2ecdcb238531a4cfdc812c7cb \
    --parts 128 --workers "$WORKERS"
python "$SLURM_SUBMIT_DIR/scripts/unzip_truncated_zip.py" "$dest/n400Data.zip" "$dest/extracted"
cd "$dest"
du -sh n400Data.zip extracted
find extracted -maxdepth 3 -type d | head -40
