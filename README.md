# MindWatch

MindWatch is research code for studying mental-health questionnaires over time.
It compares what we can learn from earlier questionnaires with what short EMA
check-ins add. EMA means a brief report of how someone feels in daily life.

This repository contains code, setup files and synthetic tests. Keep patient
records, recordings, research reports, predictions and trained weights outside
Git. No real data is needed to install the package or run its tests.

## What it predicts

The main task is to estimate **PHQ-9 ≥10 at a later assessment**, using information
available before that assessment. People enter this task when their first
observed PHQ-9 is at least 5. The first assessment provides history; it is not a
prediction target. People stay in the study if their symptoms later improve.

| Model | Information used |
| --- | --- |
| **H** | Earlier PHQ-9, GAD-7 and ISI scores, and time between assessments |
| **compact** | Previous PHQ-9 and recent EMA values for depression, anxiety, sleep and stress |

Both use regularized logistic regression. The compact model compares EMA
lookbacks of 14, 30, 60 and 90 days. These are questionnaire-based research
estimates, not clinical diagnoses.

There is also a separate EMA baseline function, `ema_forecasts`. It predicts the
next recorded EMA using either the previous response or a fixed smoothing rule
(P0). It is separate from the PHQ classifier.

## Get started

Use Python 3.10 or newer. From this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest
git config core.hooksPath .githooks
```

For Parquet input, also run `python -m pip install -e '.[parquet]'`.

## Prepare your private inputs

The commands read local CSV or Parquet files with these columns:

| Table | Required columns |
| --- | --- |
| Visits | `subject_id`, `index_date`, `phq9`, `gad7`, `isi` |
| Daily EMA | `subject_id`, `date`, `ema_depression`, `ema_anxiety`, `ema_sleep`, `ema_stress` |

Visits may also have a unique `sample_id`. Include each person's available
assessment history. Keep identifiers as strings so leading zeroes are preserved.

Use one row per person and calendar date. Dates must already be resolved to the
study's local calendar and stored at midnight, without timezone information.
Resolve duplicate responses and device clock issues before using these tools.
The code does not guess timezone conversions or sensor units.

Use NaN for missing scores, not zero. PHQ-9, GAD-7 and ISI must be valid integer
totals; EMA must already use a consistent 0–10 scale. Missing EMA fields remain
missing. Responses are not interpolated.

## Train first, then evaluate

Put each person in **one role only**: training, model selection, probability
calibration or test. Keep all of that person's visits together. If a test group
was reserved earlier, exclude those people from every development role.

The `assign_roles` helper in `mindwatch.validation` can create a reproducible
60/10/10/20 split from baseline-only rows. You can also provide your study's own
fixed participant split. This workflow evaluates new people; it does not by
itself test performance in a future calendar period.

Set these variables to private directories outside every Git checkout:

```bash
export PRIVATE_INPUT="/path/to/private-input"
export PRIVATE_OUTPUT="/path/to/private-output"

mindwatch-longitudinal train \
  --train-visits "$PRIVATE_INPUT/train_visits.csv" \
  --selection-visits "$PRIVATE_INPUT/selection_visits.csv" \
  --calibration-visits "$PRIVATE_INPUT/calibration_visits.csv" \
  --ema "$PRIVATE_INPUT/ema.csv" \
  --output "$PRIVATE_OUTPUT/new_run"

mindwatch-longitudinal evaluate \
  --test-visits "$PRIVATE_INPUT/test_visits.csv" \
  --ema "$PRIVATE_INPUT/ema.csv" \
  --run "$PRIVATE_OUTPUT/new_run"
```

Use a new output directory for each run. Training fits preprocessing on training
people, chooses a model on selection people, and saves a frozen model. Calibration
uses its own people; it stays inactive if either class has too little support.
Evaluation checks for participant overlap and does not retrain or choose again.
Load only model files you trust: joblib files can execute code when opened.

Results stay in the private output directory. They include AUROC, AUPRC, Brier
score, calibration diagnostics and threshold metrics. Each person has equal
total weight, even if some have more visits. Confidence intervals and study-specific
statistical analyses need a separate analysis protocol.

## How earlier information is kept separate

The current questionnaire supplies the answer to be evaluated, never a predictor
for that same visit. Earlier survey scores become available the next day. EMA
features use only responses before the assessment date, excluding same-day EMA.
Preprocessing is learned from training people only. A test person's own earlier
responses may be used as inputs without using that person to fit the model.

Synthetic tests check these rules, model saving/loading and the private-output
restriction. Passing them verifies software behavior, not clinical accuracy.

## The research toolbox

The broader development code is available in `mindwatch.research`. It contains
implementations and tests, without the private study files they were used with.

| Area | Modules |
| --- | --- |
| Read and check inputs | `sources`, `clinical`, `parsing`, `metadata_audit`, `raw_hourly` |
| Sensor features | `features`, `sensor_engineering`, `hourly_features`, `temporal_features`, `monitoring_data` |
| Splits and evaluation | `temporal_splits`, `monitoring_splits`, `evaluation`, `benchmarks`, `oof_sensor` |
| Models | `neural` (GRU-D, TCN and gated fusion), `mtm_adapter`, `sensor_models`, `neural_comparators`, `representations`, `stacking` |
| Weekly EMA | `ema_personalization`, `ema_sequence`, `ema_joint`, `foundation` |
| Frozen inference | `routing`, `registry` |
| Robustness and summaries | `controls`, `missingness_stress`, `analyses`, `associations`, `monitoring_report` |

Install the parts you need:

```bash
python -m pip install -e '.[test,research]'
# Optional neural models:
python -m pip install -e '.[neural]'
# Optional external MiniRocket adapter:
python -m pip install -e '.[rocket]'
```

`mindwatch-research --help` lists the research commands. Supply your own private
input paths and study configuration. For monitoring runs, `private_workspace` in
that configuration can point to a private workspace. All generated data, keys,
reports and checkpoints must stay outside Git. Reports are never copied back
into the installed package. The C++17 hourly parser needs `clang++` or `g++`.

GRU-D, TCN and the EMA sequence comparator train from scratch on the supplied
training data. Masked representations use only supplied training windows and
rotate one eligible window per person per epoch. Split creation and feature
selection must still happen inside the study's development procedure. The
sequence comparator documents its own-training residual baseline; `oof_sensor`
provides the separate time-OOF correction path.

MTM and MiniRocket use explicitly prepared, hash-verified upstream source kept
outside Git. Upstream licenses apply; their implementations are not redistributed
here. MiniRocket setup is available through
`mindwatch.research.minirocket_adapter.prepare_source`. Chronos-2 and Timestral
wrappers accept already prepared local model directories. Their weights and
optional runtime packages are not included, and fitting does not download them.
To run MTM integration tests against an already verified external checkout, set
`MINDWATCH_MTM_SOURCE` to its directory before running pytest. Tests do not fetch
upstream source or weights automatically.

The `routing` and `registry` APIs replace study-specific interfaces tied to
private fitted models and participant lists. Create a new private registry from
your own fitted predictors; no active clinical model is bundled. Historical
report scripts, study-specific launchers and fitted manifests remain outside
this repository. Their reusable methods are exposed through the modules above.
The private research workspace is not synchronized wholesale.

Local speech-processing code is also available as `mindwatch.research.speech`
with the optional `speech` dependencies. Consent, task, speaker and review gates
must be supplied explicitly. It remains separate from the H/compact workflow.

## Older scripts and contributions

Earlier sensor, voice, tabular and TFT scripts remain in `src/`. Their optional
dependencies are listed in `requirements.txt`. Their training entry points now
stop before loading inputs. The audit found cohort-wide imputation and model
selection paths that do not satisfy the current validation rules. Their source
is retained for reference, not for new training or performance claims. Use
`mindwatch-longitudinal` or the explicitly configured research tools instead.
The H and compact classifiers do not use sensor or voice inputs.

Run older scripts from a private working directory. Set `MINDWATCH_CONFIG` to a
private YAML file; the filenames in `config.yaml` are placeholders. This applies
to the remaining descriptive helpers, not to the retired training commands.

New runs reject malformed dates, non-binary classification targets and invalid
training schedules. Calibration requires distinct participants in both classes.
Unsupervised EMA output heads fall back to the supplied baseline. Hourly input
must contain valid local timestamps without a timezone suffix; no conversion is
guessed. Speech inventories include source fingerprints and must be rebuilt if
the source archives change.

Model registries record a source fingerprint. A code update requires a new
training/freeze run; keep older models with their matching code version. The
software checks do not replace re-evaluation on a study's locked data.

Before committing, run `python tools/check_code_only.py --staged` and review the
diff. Git hooks and CI reject files outside the code allowlist, including data,
reports, logs and weights. Tests generate synthetic records at runtime, and CI
does not upload artifacts. These checks help catch mistakes but cannot recognize
every sensitive value. Never paste real records into issues, PRs or CI logs.

The default branch contains code only. Keep backups of earlier research
workspaces private; do not merge their Git history into this repository.

See [LICENSE](LICENSE) for reuse terms.
