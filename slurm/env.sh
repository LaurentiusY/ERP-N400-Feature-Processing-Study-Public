# Environment shared by every FIR job. Source it; do not run it.
#
# Required:  export N400_ROOT=<persistent project folder>, e.g.
#            export N400_ROOT=~/projects/def-<PI>/$USER/n400
# Optional overrides: N400_REPO, N400_VENV, N400_DATA_DIR, N400_DERIV_DIR, N400_RESULTS_DIR,
# N400_DRASCHKOW_DIR, N400_PURSUE_DIR, N400_MODULES, N400_R_MODULE, R_LIBS_USER.
# Dataset (guide sections 13, 14): N400_DATASET=erpcore|scene|sentence|pursue (default erpcore); sbatch
# passes the submitting shell's environment, e.g. `N400_DATASET=scene sbatch ...`.

: "${N400_ROOT:?export N400_ROOT first, e.g. ~/projects/def-<PI>/\$USER/n400}"
export N400_REPO="${N400_REPO:-$N400_ROOT/ERP-N400-Feature-Processing-Study}"
export N400_VENV="${N400_VENV:-$N400_ROOT/venv}"
export N400_DATA_DIR="${N400_DATA_DIR:-$N400_ROOT/data}"
export N400_DERIV_DIR="${N400_DERIV_DIR:-$N400_ROOT/derivatives}"
export N400_RESULTS_DIR="${N400_RESULTS_DIR:-$N400_ROOT/results}"
export N400_DRASCHKOW_DIR="${N400_DRASCHKOW_DIR:-$N400_ROOT/external/draschkow2018/extracted/n3n4}"
export N400_PURSUE_DIR="${N400_PURSUE_DIR:-$N400_ROOT/external/pursue}"
export N400_DATASET="${N400_DATASET:-erpcore}"
# Preprocessing variant (guide 13.11): main (default) or prespecified (scene, sentence only)
export N400_PREPROC="${N400_PREPROC:-main}"
export MNE_DONTWRITE_HOME=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # parallelism via joblib
# R (exploratory pairwise comparisons only): module loaded by slurm/run_pairwise.sh, packages
# afex and emmeans installed once by slurm/setup_r.sh into this library
export N400_R_MODULE="${N400_R_MODULE:-r}"
export R_LIBS_USER="${R_LIBS_USER:-$N400_ROOT/Rlib}"

# Module names: check with `module spider python` / `module spider arrow` on FIR.
# pyarrow comes from the arrow module on Alliance clusters (not from pip).
N400_MODULES="${N400_MODULES:-StdEnv/2023 python/3.11 arrow}"

case $- in *u*) _n400_nounset=1 ;; *) _n400_nounset= ;; esac
set +u
# shellcheck disable=SC2086
module load $N400_MODULES
if [ -f "$N400_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$N400_VENV/bin/activate"
fi
[ -n "$_n400_nounset" ] && set -u
unset _n400_nounset
