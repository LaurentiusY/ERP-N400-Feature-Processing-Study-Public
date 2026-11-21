# Feature processing for single-trial semantic decoding from EEG

Code, pre-registration and participant-level results for

> Yang, L., & Lankarany, M. *Feature processing for single-trial semantic decoding from EEG: a
> factorial comparison with pre-registered replication.* Preprint, bioRxiv (link to follow).

The study crosses four feature-processing choices (signal representation, time window,
normalization and channel set; 640 configurations) and decodes, from single EEG trials, whether a
stimulus fitted its semantic context in three tasks: visual word pairs (ERP CORE N400), objects in
visual scenes and words in spoken sentences (Draschkow et al., 2018), 40 participants per task.
Eleven planned comparisons are tested jointly over the three tasks, and the two most robust
conclusions are replicated in a pre-registered study of 283 participants from the PURSUE N400
dataset.

## Contents

| Path | Content |
|---|---|
| `src/n400fp/` | Python package: preprocessing, feature extraction, decoding, statistics (`python -m n400fp ...`) |
| `scripts/` | Sensitivity analyses, PURSUE replication, data download helpers, and the scripts that make the figures and supplementary tables of the paper |
| `slurm/` | Job scripts for a Slurm cluster (the analyses were run on the Alliance cluster FIR) |
| `tests/` | Unit and end-to-end tests on synthetic data (`pytest`) |
| `preregistration/` | Pre-registration of the PURSUE replication: Chinese original and English translation |
| `results/` | Participant-level decoding results and the group-level reports used in the paper |
| `requirements-lock.txt` | Exact package versions of the environment the analyses ran in |

## Install

Python 3.11 (3.12 also works).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                 # core + tests
pip install -e ".[download]"                            # optional: ERP CORE download helper
# EEGNet (optional): install torch for your machine first, then
pip install -e ".[eegnet]"
```

`requirements-lock.txt` records the exact versions used for the paper (Python 3.11.5, Alliance
cluster wheels). To get close to that environment, install from it before installing the
package: `pip install -r requirements-lock.txt && pip install -e . --no-deps`. A few entries are
cluster builds (for example `torch`, `pillow_simd`) and may need the matching PyPI build.

## Reproduce the figures and supplementary tables

The results in `results/` are enough to regenerate every figure and supplementary table of the
paper, without the EEG data:

```bash
python scripts/paper_figures.py       # Figures 1-7  -> paper/NeuroImage/figures/
python scripts/paper_supplement.py    # Tables S1-S9 -> paper/NeuroImage/supplement/tables/
```

## Results

The layout follows the analysis code: ERP CORE (word pairs) at the top level of `results/`,
the other tasks in subfolders named after the dataset (`scene/`, `sentence/`, `pursue/`).

| File | Content |
|---|---|
| `[<task>/]reports/full/subject_scores.csv` | Balanced accuracy of every participant, configuration and model (mean over 10 outer folds x 5 repeats) |
| `[<task>/]reports/full/marginal_tables/lr_*.csv` | Participant-level marginal means of each factor (logistic regression) |
| `[<task>/]reports/full/planned_contrasts_*.csv`, `rm_anova_lr.csv`, `art_anova_lr.csv` | Planned comparisons P1-P11 and ANOVA per task |
| `[<task>/]reports/full/timecourse.csv`, `exploratory/` | Time-resolved decoding; condition effects and correlations |
| `[<task>/]reports/preprocessing/preproc_summary.csv` | Trials kept, ICA components removed and interpolation per participant |
| `[<task>/]eegnet/sub-XXX.parquet` | EEGNet accuracy per participant, configuration, seed and fold |
| `joint/` | Joint test of P1-P11 over the three tasks |
| `pursue/reports/` | Replication: tests, results per site, participants included, preprocessing |
| `c_grid_sensitivity/`, `preproc_sensitivity/`, `tfr_sensitivity/` | Sensitivity analyses |

**Configuration codes** (`B1-W1-N2-C1` is the baseline):

- **B**, representation: B1 waveform; B2-B5 theta, alpha, beta and low-gamma band envelopes; B6
  B2-B5 concatenated; B7 relative band amplitudes; B8 B1 and B2-B5 concatenated.
- **W**, time window: W1 200-800 ms; W2 300-500 ms; W3 500-800 ms; W4 0-800 ms.
- **N**, normalization: N1-N4 (defined in the Methods of the paper).
- **C**, channel set: C1-C5 (defined in the Methods of the paper).

Fold-level results of the main decoding are not included (several thousand files per task);
they can be regenerated with the code below.

## Rerun the analyses from the EEG data

The data are openly available: ERP CORE (https://doi.org/10.18115/D5JW4R), Draschkow et al.
(2018) (https://doi.org/10.5281/zenodo.1421347) and PURSUE (https://doi.org/10.82901/nemar.on007052).
Folders are set with environment variables (defaults inside the repository):
`N400_DATA_DIR` (`data/`), `N400_DERIV_DIR` (`derivatives/`), `N400_RESULTS_DIR` (`results/`),
`N400_DRASCHKOW_DIR` and `N400_PURSUE_DIR`. The dataset is chosen with `--dataset` or
`N400_DATASET` (`erpcore`, `scene`, `sentence`, `pursue`). Rerunning writes into the results
folder; point `N400_RESULTS_DIR` elsewhere to keep the included results.

The analyses were run as Slurm job arrays (one task per participant). On a Slurm cluster:

```bash
export N400_ROOT=<project folder>                      # data, derivatives and results go here
export N400_REPO=$N400_ROOT/ERP-N400-Feature-Processing-Study-Public
bash slurm/setup.sh                                    # virtualenv (Alliance wheelhouse)
source slurm/env.sh && mkdir -p logs

# ERP CORE
sbatch slurm/download.sbatch
sbatch --array=0-39 slurm/preprocess.sbatch            # ICA, autoreject, band envelopes
sbatch --array=0-39 --time=12:00:00 --cpus-per-task=16 slurm/decode.sbatch full
sbatch --array=0-39 slurm/timecourse.sbatch
sbatch slurm/stats.sbatch full                         # -> results/reports/full/
sbatch --array=0-39 slurm/eegnet.sbatch "$N400_RESULTS_DIR/reports/full/eegnet_configs.txt"

# Draschkow et al. (2018): the same steps with N400_DATASET=scene and N400_DATASET=sentence
sbatch slurm/download_draschkow.sbatch
python scripts/check_draschkow_timing.py               # event times against the authors' epochs

# Joint test of the three tasks, after `stats --stage full` for each task
python -m n400fp joint

# PURSUE replication (288 recordings; three without EEG data are skipped)
export N400_DATASET=pursue
sbatch slurm/download_pursue.sbatch
sbatch --array=0-287 slurm/preprocess.sbatch
sbatch --array=0-287 --time=2:00:00 slurm/decode.sbatch replication
python -m n400fp summary && python scripts/pursue_replication.py
```

Sensitivity analyses (after the main study):

- **C grid**: `decode.sbatch full lr extended` for each task, then `scripts/c_grid_sensitivity.py`.
- **Time-frequency**: `slurm/tfr.sbatch`, then `decode.sbatch tfr lr` for each task, then
  `scripts/tfr_sensitivity.py`.
- **Preprocessing** (scene and sentence): rerun preprocessing and `decode.sbatch full` with
  `N400_PREPROC=prespecified`, then `stats --stage full` and `scripts/preproc_sensitivity.py`.

Without Slurm, the same steps run with `python -m n400fp <command>` (`download`, `preprocess`,
`bands`, `tfr`, `summary`, `decode`, `permutation`, `timecourse`, `eegnet`, `stats`, `joint`; see
`python -m n400fp --help`). Seeds are fixed (outer folds 20261011 plus the participant number),
so a rerun gives the same folds.

## Tests

```bash
pytest                      # fast unit tests on synthetic data
pytest -m slow              # end-to-end: synthetic recording -> ICA -> autoreject -> decoding
pytest -m eegnet            # only with torch and braindecode installed
```

## Notes

- Comments in the code refer to sections of the study's internal design document ("guide
  section 13.8" etc.); the same content is in the Methods of the paper, and section 14 is the
  pre-registration in `preregistration/`.
- `slurm/stats.sbatch full` also tries an exploratory R step that is not part of the paper and
  not included here; it prints "pairwise comparisons skipped" and the job continues.
- Parts of the preprocessing follow the code of Kessler et al. (2025)
  (https://github.com/kesslerr/m4d, CC BY 4.0).

## License

The code is released under the MIT License (see `LICENSE`). The EEG datasets keep their own
licenses: see the dataset pages above.
