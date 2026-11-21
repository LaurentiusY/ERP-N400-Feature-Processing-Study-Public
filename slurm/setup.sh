#!/bin/bash
# One-time setup, run on a FIR LOGIN node (it needs internet for PyPI). From the repository
# root:
#   export N400_ROOT=~/projects/def-<PI>/$USER/n400
#   bash slurm/setup.sh
# Re-running it is safe (it reuses the virtualenv and reinstalls the project in place).
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$N400_DATA_DIR" "$N400_DERIV_DIR" "$N400_RESULTS_DIR" "$N400_REPO/logs"

if [ ! -f "$N400_VENV/bin/activate" ]; then
    virtualenv --no-download "$N400_VENV"
fi
source "$N400_VENV/bin/activate"
pip install --no-index --upgrade pip
# Alliance-built torch (matches the cluster's CUDA driver) for the EEGNet check; install it
# before the project so pip does not fetch a PyPI build.
pip install --no-index torch
# The rest from the Alliance wheelhouse where available, PyPI otherwise.
pip install -e "$N400_REPO[dev,download,eegnet]"

python - <<'PY'
import importlib

for name in ("numpy", "scipy", "pandas", "pyarrow", "sklearn", "mne", "autoreject", "picard",
             "torch", "braindecode", "skorch", "pipeline"):
    module = importlib.import_module(name)
    print(f"{name:12s} {getattr(module, '__version__', '?')}")
PY
echo "setup done"
