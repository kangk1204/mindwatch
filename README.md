# MindWatch research code

Reusable code for longitudinal mental-health research. This repository distributes
source code and software configuration only. It does not distribute datasets,
analysis reports, individual predictions, recordings, embeddings, fitted weights,
or clinical performance results. The existing [license](LICENSE) applies.

## Install the lightweight longitudinal tools

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest
git config core.hooksPath .githooks
python tools/check_code_only.py
```

Parquet input additionally requires `pip install -e '.[parquet]'`. The original
standalone sensor, voice, tabular and TFT scripts remain under `src/`; their
optional dependencies are in `requirements.txt`. They are legacy research entry
points, separate from the tested longitudinal package. The new tests do not
constitute a scientific revalidation of those older pipelines. Voice is not an
input to the longitudinal classifiers below.

## Input contract

Keep all input files and output directories outside every Git checkout. Use
local CSV or Parquet files; do not commit example patient records.

* Visits: `subject_id`, `index_date`, `phq9`, `gad7`, `isi`, and optional unique
  `sample_id`. Supply every available historical visit for each assigned person.
* Daily EMA: `subject_id`, `date`, `ema_depression`, `ema_anxiety`, `ema_sleep`,
  `ema_stress`. Missing domains use NaN. Each domain must already be harmonized
  to 0–10; no direction or unit is guessed.
* Keys must be nonmissing. There must be at most one row per person/calendar
  date in each table. Resolve duplicates and same-day repeat responses upstream.
* Dates must be local, timezone-resolved calendar dates at midnight. This code
  neither guesses UTC/KST nor resolves device clocks or raw export units.
* Surveys use validated score ranges. Missing scores remain missing. Zero is a
  score, not an inferred nonwear code. Raw device parsing needs its own audit.

The current PHQ task includes people whose first observed PHQ is at least 5,
keeps them after improvement, and predicts **current PHQ>=10 at later visits**.
This is a questionnaire-defined screening outcome, not a diagnosis. The first
visit supplies history and is not a prediction target. Every later visit can be
a sample, while all samples from one person remain in a single population role.

Current scores are attached as labels after feature construction. Survey history
is available from the next calendar day. EMA uses only actual observations in
`[visit date - lookback, visit date)`; same-day responses are excluded. No future
observations, interpolated EMA, participant identifiers or current questionnaires
are passed to a classifier. Own past history remains available for a new person
whose records have never been used for population fitting.

## Models and separate execution stages

* `H`: 22 historical survey features, median imputation with missing indicators,
  standard scaling and regularized logistic regression (`C=0.01`).
* `compact`: previous PHQ plus each EMA domain's last value, median and age.
  Train-only observation/variation filtering, median imputation, person-weighted
  scaling, clipping and logistic regression (`C=1`).
* Candidate EMA windows are fixed to 14, 30, 60 and 90 days. Select on raw
  validation AUROC, with exact ties resolved by Brier, H, then shorter window.
  This limited candidate search does not establish an optimal model.

Assign **disjoint people** to training, selection, calibration and test roles
before running. `mindwatch.validation.assign_roles` accepts baseline-only rows
and provides a deterministic 60/10/10/20 helper split. Existing reserved test
people must first be excluded from development. The training command has no test
input. It fits preprocessing only on training people, chooses on selection
people, and freezes models before the separate evaluation command can run.

```bash
mindwatch-longitudinal train \
  --train-visits "$PRIVATE_INPUT/train_visits.csv" \
  --selection-visits "$PRIVATE_INPUT/selection_visits.csv" \
  --calibration-visits "$PRIVATE_INPUT/calibration_visits.csv" \
  --ema "$PRIVATE_INPUT/ema.csv" --output "$PRIVATE_OUTPUT/new_run"

mindwatch-longitudinal evaluate \
  --test-visits "$PRIVATE_INPUT/test_visits.csv" \
  --ema "$PRIVATE_INPUT/ema.csv" --run "$PRIVATE_OUTPUT/new_run"
```

Set `PRIVATE_INPUT` and `PRIVATE_OUTPUT` to approved private directories outside
Git. The output run must not already exist. Artifacts use restricted filesystem
permissions. Freeze records bind the model and source checksums. Evaluation
rejects overlap with all development roles and never refits or reselects. Use
only trusted local model files: joblib loading can execute code.

Each participant has equal total evaluation/training weight. Offset-only
calibration and a target-sensitivity threshold use separate calibration people;
insufficient class support leaves calibration inactive. Evaluation computes raw
and calibrated AUROC, AUPRC, Brier, diagnostic calibration intercept/slope and
threshold metrics. The diagnostic calibration fit never changes predictions.
This lightweight path does not implement confidence intervals or prospective
calendar validation. Clinical or publication claims require the study-specific
protocol and additional statistical evaluation.

`mindwatch.features.ema_forecasts` is a **separate next-response EMA task** with
previous-value and fixed P0 baselines. P0 averages the previous EMA and the median
of up to three past EMA responses. It does not turn the PHQ classifier into an
EMA forecasting model.

## Confidentiality and contribution boundary

`.gitignore`, local Git hooks, and a CI tracked-file checker reject datasets,
models, analysis documents, logs, notebooks, binaries and private paths. Tests
generate synthetic records at runtime in temporary directories; no record files
are committed. The checker is a technical guard, not a guarantee against every
secret or reidentification risk. Review the complete staged diff before pushing.
Do not upload real data to CI, issues, pull requests, external inference APIs or
test artifacts. CI does not upload artifacts.

Legacy scripts can still create local outputs. Run them in an approved private
working directory with `MINDWATCH_CONFIG` pointing to a private YAML file. Do not
commit that file. Example filenames in `config.yaml` are placeholders.

Previously tracked run logs are removed from the current tree by this change;
removal does not erase older Git history. This change does not rewrite history.
