#!/bin/bash
# The test suite in the FIR environment (no data needed): fast, slow (ICA, autoreject on a
# synthetic recording) and EEGNet (on CPU). Every group runs even if an earlier one fails.
#   sbatch slurm/tests.sbatch
#SBATCH --job-name=n400-tests
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
cd "$SLURM_SUBMIT_DIR"
status=0
for marker in "not slow and not eegnet" "slow" "eegnet"; do
    echo "=== pytest -m \"$marker\""
    python -m pytest -p no:cacheprovider -q -m "$marker" || status=1
done
exit $status
