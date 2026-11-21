# PURSUE replication: pre-registration (English translation)

**This is a translation.** The binding text is the Chinese original in
[`pursue_preregistration_zh.md`](pursue_preregistration_zh.md), copied verbatim from section 14.1–14.8
of the study's internal engineering guide at commit `d2a8f71`, tagged `prereg-pursue-v1` on
2026-10-15 15:27 (UTC−4). The PURSUE EEG data were downloaded after this commit. The internal
repository and its history are not public.

**Section numbers** refer to the internal guide: sections 4 and 5 are the fixed ERP CORE
preprocessing and the band envelopes, section 6 the feature processing, decoding and statistics,
and section 13 the joint study of the three tasks (13.5 the joint test, 13.8 the preprocessing
changes for the Draschkow et al. tasks). All of these are described in the Methods of the paper.
Configuration codes (`B1-W1-N2-C1` etc.) and the planned comparisons P1–P11 are defined there as
well. Deviations from this plan are reported in the Methods of the paper.

---

## 14. Pre-registered replication on PURSUE (version 7, 2026-10-15)

### 14.1 Aim and role

A **pre-registered direct replication**, on an independent large sample, of the two most robust
conclusions of the joint test of the three tasks. The replication is a supplementary result: it
does not enter the joint test of section 13.5 and does not change the existing conclusions. The
researcher had earlier (2026-10-12) ruled out PURSUE; after the study of the three datasets was
completed, this small-scale replication was added instead (2026-10-15). Only two conclusions are
tested; the reason is given in section 14.5.

### 14.2 Data

- **Source**: PURSUE N400 Word Processing (Couperus, Bukach & Reed), NEMAR `on007052` version
  v1.0.0, i.e. OpenNeuro `ds007052` (DOI 10.82901/nemar.on007052), CC0. Recorded in 2017–2018 at
  three primarily undergraduate institutions (Mt. Holyoke College, University of Richmond,
  Claremont McKenna College). Participant codes fall into three groups: `sub-1xxx` 104 people,
  `sub-2xxx` 78, `sub-3xxx` 106, 288 in total, unrelated to the ERP CORE and Draschkow
  participants.
- **Task**: the ERP CORE N400 word-pair paradigm: prime → target → button press (related /
  unrelated). Event codes as in ERP CORE: primes 111/112/121/122, related targets 211/212,
  unrelated targets 221/222, responses 201/202; practice trials 110/120/210/220 (event description
  "Unknown") are not included. 120 targets per participant are expected (60 per class).
- **Recording**: BrainVision actiCHamp, 500 Hz, 29 EEG electrodes (FP1, Fz, F3, F7, FC3, C3, C5,
  TP9, CPz, Pz, P3, P7, PO3, PO7, O1, Oz, FP2, F4, F8, FC4, Cz, C4, C6, TP10, P4, P8, PO4, PO8, O2)
  and 3 EOG channels (HEOG R, HEOG L, VEOG Lower), ground AFz; the online reference is not stated
  in the metadata ("common"). There is no FCz and no P9, P10. BIDS format, one EEGLAB `.set` per
  participant (data embedded).
- **Download and checks**: downloaded on FIR file by file from the NEMAR file list
  (`manifest.json`); `.set` files are checked against the sha256 in the list, the other files
  against their size.
- **Seen before registration** (no EEG signal values): the dataset description, README,
  `dataset_description.json`, `task-WordPR_events.json`, `participants.tsv`, the file list (with
  file sizes), the authors' `code/` files (channel operations for the linked-mastoid reference and
  the EOG derivations; ERPLAB bin files), and the channel table, event table and `.set` file
  header of sub-1001.

### 14.3 Preprocessing

The fixed ERP CORE path of sections 4 and 5 is used (Kessler's Luck reference path, with the two
deviations of version 5); the changes of section 13.8 for the Draschkow tasks are not used. Only
the following necessary adaptations are made:

| Step of section 4 | ERP CORE | PURSUE |
|---|---|---|
| 1. Reading and events | EEGLAB; targets + 26 ms | EEGLAB (BIDS); targets by the same event codes, **no display delay added** (the dataset reports no monitor delay; the 26 ms of ERP CORE is the measured value of its monitor) |
| 2. Downsampling | 1024 → 256 Hz | 500 → 256 Hz |
| 3. EOG channels | HEOG = left − right; VEOG = lower − FP2 | HEOG = HEOG L − HEOG R; VEOG = VEOG Lower − FP2 (consistent with the authors' channel operations) |
| 4. Initial reference | Cz | Cz |
| 9. Final reference | mean of P9, P10 | mean of TP9, TP10 (the authors' offline reference; both lie low behind the ears) |

- **Channels**: saving and features use the 27 of the 28 ERP CORE recording electrodes that are
  present (FCz is missing; FCz is not reconstructed, because the online reference electrode
  cannot be verified). Like P9, P10 in ERP CORE, TP9 and TP10 do not enter any channel set. This
  replication uses only C1.
- **Trials**: all target trials enter decoding, without selection by response accuracy or
  reaction time (as in ERP CORE).
- Envelopes are computed as in section 5 (needed for B2, B8).

### 14.4 Decoding

- 3 configurations: `B1-W1-N2-C1` (baseline), `B8-W1-N2-C1`, `B2-W1-N2-C1`.
- Main model: L2 logistic regression, C chosen from the pre-specified grid {10⁻³, …, 100} by
  inner 5-fold cross-validation; outer 10 folds × 5 repeats (seed 20261011 plus the participant
  number), balanced accuracy; as in section 6.
- Secondary: shrinkage LDA on the same 3 configurations (N2 replaced by the equivalent N1),
  reported only.

### 14.5 Hypotheses, tests and replication criterion

For each participant, the mean balanced accuracy over the 50 outer folds is computed:

| Code | Prediction (one-sided) | Corresponding conclusion of the joint test | Condition effect in the existing data (logistic regression, ERP CORE / scenes / sentences) |
|---|---|---|---|
| **R1** | `B8-W1-N2-C1` < `B1-W1-N2-C1`: adding band envelopes to the waveform lowers accuracy | P1 (general) | −0.037 / −0.035 / −0.028 (dz −0.64 / −0.64 / −0.53) |
| **R2** | `B1-W1-N2-C1` > `B2-W1-N2-C1`: the waveform beats the theta envelope | P3 (same direction) | +0.135 / +0.057 / +0.066 (dz 1.50 / 0.85 / 0.95) |

- Test: one-sided paired t-test on the participant-level differences, R1 and R2 Holm-corrected
  together, α = 0.05; Wilcoxon signed-rank test as a check. The difference, its 95% confidence
  interval and dz are reported.
- **Replication succeeds** if R1 and R2 are both significant after Holm correction and in the
  predicted direction. If either one is not significant or goes in the opposite direction, that
  one is reported as not replicated.
- **Manipulation check**: balanced accuracy of the baseline configuration above 0.5 (one-sided
  t-test), reported only.
- Power: the smallest condition effect in the existing data is dz = 0.53; with about 280
  participants, power at one-sided α = 0.05 is > 0.999.
- No other comparison is tested. P7 (trial-wise z-score) holds generally in the marginal means,
  but its condition effect at the baseline levels is small and inconsistent in direction
  (ERP CORE −0.005, scenes +0.001, sentences −0.004, none significant), so its direction cannot be
  predicted reliably and it is not registered. The replication tests "the effect with the other
  dimensions at their baseline levels", which is not exactly the marginal mean used in the joint
  test; the paper reports it as a supplementary replication.

### 14.6 Exclusion criteria

- **Excluded in advance**: recordings whose `.set` file contains no EEG data (the file is only
  19 KB: sub-2002, sub-2025, sub-3042).
- **Excluded after processing**: recordings that cannot be read; participants with fewer than
  30 trials in either class after preprocessing.
- No participant is excluded on decoding accuracy or any other outcome. The numbers included and
  excluded at each institution are reported, as are the R1 and R2 results without the
  post-processing exclusion.

### 14.7 Secondary and exploratory results

- R1 and R2 at each institution separately (descriptive); the LDA results.
- Participant codes: mapped in BIDS order to sub-001 to sub-288; the mapping table is saved.
- Any implementation detail that differs from this section is recorded as a deviation after
  implementation, with the reason.

### 14.8 How the plan is fixed

The PURSUE EEG data are downloaded on FIR only after this section has been merged into main and
tagged `prereg-pursue-v1`; after that, 14.2–14.6 are not changed. The implementation code is
committed after the tag, and the implementation must not change the preprocessing,
configurations, tests and exclusion criteria specified in this section.
