# MindWatch (Tabular AI for Mental Health)

> This code is part of the project “Development of a Personalized Digital Mental Health Care Platform Combining Robotics, AI, and Community Services” (PI: Jung Jae Lee, Dankook University).

MindWatch converts multi-modal wearables, voice clips, and survey responses into publication-ready predictive models for depressive symptoms. The pipeline is built around leakage-safe feature engineering, automated hyperparameter tuning, and rich evaluation outputs (plots, tables, JSON summaries) that you can drop directly into papers.

## What's inside

- **Leakage-safe feature builder** (`src/run_tabular_models.py`)  
  Rolling statistics, cross-wave deltas, ratios, EWM trends, text encodings, and voice descriptors are created in one controlled place with participant-level splits to prevent data leakage. Text features automatically drop PHQ/GAD/DSM/Loneliness responses to avoid questionnaire leakage.
- **Automation-ready pipeline** (`src/run_full_pipeline.py`)  
  Run tuning → evaluation → reporting with a single command; generates ROC/PR/calibration plots and publication tables automatically.
- **Optuna tuning CLI** (`src/tune_tabular_models.py`)  
  Supports CPU/GPU tuning, persistent SQLite studies, time-block validation, and sampled feature budgets.
- **Rich evaluation suite** (`src/evaluate_tabular_strategy.py`)  
  Outputs ROC/PR/calibration plots, confusion matrices, best-F1/F2 thresholds, and all metrics in JSON for reproducibility.
- **Tabular deep learning baselines** (`src/train_tabular_dl.py`)  
  MLP/TabNet training on the engineered feature set for comparison against tree-based methods.
- **Multi-modal fusion toolkit** (`src/run_full_pipeline.py`)  
  Sensor-only, text-only, voice-only branches plus late-fusion reporting, parallel strategy evaluation (`--n-eval-jobs`), and optional DL benchmarking via `--run-tabular-dl`.
- **Explainability toolkit** (`src/generate_feature_explanations.py`)  
  Rebuilds the tuned XGB model, computes permutation SHAP values on a leakage-safe holdout, and exports ranked TSV + SVG plots with Hangul-safe fonts.
- **Publication helpers** (`src/build_publication_tables.py`)  
  Converts pipeline results into ready-to-use CSV/Markdown tables.

## Repository layout

```
00_input_data/          # hourly sensor data (user supplied, Git ignored)
00_input_voice_data/    # raw mp3 clips (user supplied, Git ignored)
00_label_data/          # survey sheets (user supplied, Git ignored)
logs/                   # Optuna studies, evaluation JSON, summary tables (ignored)
results/                # pipeline outputs (ignored)
plots/                  # ROC / PR / calibration charts (ignored)
src/
 ├─ run_full_pipeline.py         # one-click tuning + evaluation + reporting
 ├─ run_tabular_models.py        # feature builder + baseline sweep + stacking
 ├─ tune_tabular_models.py       # Optuna CLI for individual strategies
 ├─ evaluate_tabular_strategy.py # detailed evaluation + plots
 ├─ run_optuna_batch.py          # helper to run multiple tuning jobs sequentially
 ├─ build_publication_tables.py  # turns pipeline results into CSV/Markdown tables
 ├─ train_tft.py                 # shared data-loading utilities for TFT
 ├─ train_tabular_dl.py          # MLP / TabNet baselines
 └─ generate_feature_explanations.py # SHAP-style feature importance TSV + SVGs
```

## Prerequisites

- Python 3.10+ (tested on 3.12)
- Ubuntu 22.04 LTS or macOS 13+ (Apple Silicon works; GPU tuning not supported there)
- Optional CUDA GPU for XGBoost/LightGBM tuning

### Installation

```bash
# Clone and enter the repo
git clone git@github.com:<YOUR_USERNAME>/mindwatch.git
cd mindwatch

# (Ubuntu) system dependencies
sudo apt-get update && sudo apt-get install -y build-essential python3-venv

# (macOS) ensure Command Line Tools are installed
xcode-select --install

# Create and activate virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Data**  
> - Place hourly sensor CSVs in `00_input_data/` and survey/label workbooks in `00_label_data/`.  
> - Drop per-participant voice folders in `00_input_voice_data/` (e.g., `participant_id/*.mp3`).  
> - All of these directories are Git ignored by default.

## Quickstart: end-to-end pipeline

The easiest way to train, tune, and collect paper-ready outputs is:

```bash
source .venv/bin/activate

python src/run_full_pipeline.py \
  --run-tuning \
  --history-hours 240 \
  --cv-folds 5 \
  --top-k-features 120 \
  --top-k-min 40 \
  --tuning-trials 50 \
  --use-gpu \
  --block-validation \
  --study-name full_pipeline_voice_text_v2 \
  --output-dir results/YYYYMMDD_run1
```

Optional but recommended (runs faster once caches exist):

```bash
PYTHONPATH=src python - <<'PY'
from train_tft import load_label_frames
from voice_features import build_voice_feature_table
from text_features import build_text_feature_table
labels = load_label_frames()
build_voice_feature_table(labels)
build_text_feature_table(labels)
PY

# After the pipeline, regenerate SHAP TSV + SVG plots (200-sample holdout)
PYTHONPATH=src python src/generate_feature_explanations.py \
  --optuna-json results/YYYYMMDD_run1/optuna_xgb_full_<timestamp>.json \
  --output-tsv results/YYYYMMDD_run1/shap_feature_importance.tsv \
  --output-bar plots/shap_bar_top_features.svg \
  --output-beeswarm plots/shap_beeswarm.svg

# (Optional) Re-run pipeline with late-fusion reports and DL baseline
python src/run_full_pipeline.py \
  --history-hours 240 \
  --cv-folds 5 \
  --top-k-features 120 \
  --top-k-min 40 \
  --n-eval-jobs 4 \
  --strategies hgb_top120 xgb_full \
  --run-tabular-dl \
  --output-dir results/YYYYMMDD_run1_fusion
```

This command:
1. Runs Optuna tuning for HGB/LightGBM/XGB (reusing existing studies if present).
2. Evaluates tuned strategies + stacking meta-ensemble.
3. Saves:
   - `results/.../full_pipeline_results_<timestamp>.txt`: holdout/CV metrics + stacking results
   - `results/.../publication_table_metrics.(csv|md)`: ready-to-use performance tables
   - ROC/PR/calibration plots in `plots/`
   - Optuna JSON summaries + SQLite DB in `results/.../`

### Need only evaluation tables later?
```
python src/build_publication_tables.py --log-path results/<run>/full_pipeline_results_*.txt
```

## Manual workflow (advanced)

1. **Baseline sweep**  
   ```bash
   python src/run_tabular_models.py --history-hours 240 --cv-folds 5
   ```
   Produces `logs/tabular_results.txt`, stacking metrics, and holdout summaries.

2. **Strategy tuning** (single family)  
   ```bash
   python src/tune_tabular_models.py \
     --strategy hgb_top120 \
     --history-hours 240 \
     --cv-folds 5 \
     --trials 200 \
     --top-k-features 120 \
     --top-k-min 40 \
     --block-validation \
     --use-gpu \
     --study-name hgb_tuning \
     --storage sqlite:///logs/optuna_hgb.db
   ```
   Generates `logs/optuna_hgb_top120_<timestamp>.json` + SQLite study.

3. **Detailed evaluation + plots**  
   ```bash
   python src/evaluate_tabular_strategy.py \
     --strategy XGB_v5_full \
     --history-hours 240 \
     --cv-folds 5 \
     --top-k-features 120 \
     --top-k-min 40 \
     --output-prefix XGB_best
   ```
   Outputs:
   - `plots/roc_XGB_best.png`
   - `plots/pr_XGB_best.png`
   - `plots/calibration_XGB_best.png`
   - `logs/evaluation_XGB_best.json` (ROC/PR data, calibration curve, confusion matrices, best-F1/F2 thresholds, default threshold metrics)

4. **Tabular deep learning baselines**  
  ```bash
  python src/train_tabular_dl.py \
    --model-type mlp \
    --history-hours 240 \
    --cv-folds 5 \
    --top-k-features 120 \
    --top-k-min 40 \
    --epochs 50 \
    --batch-size 256
   ```
   Results land in `logs/tabular_<model>_<scenario>_<timestamp>.json`.

## GPU notes

- **XGBoost**: `--use-gpu` switches to `tree_method="gpu_hist"`. If CUDA is not available it automatically falls back to CPU `hist`.
- **LightGBM**: Requires a GPU-enabled build; otherwise the warning is suppressed and CPU training continues.
- **CatBoost**: CPU by default; manually add `task_type=GPU` if your environment supports it.

## How the pipeline works

**1) Data in**  
- Sensor CSVs (hourly or daily) in `00_input_data/`; daily signals are shifted +1 day to avoid look-ahead.  
- Voice clips (`00_input_voice_data/`), each mapped to the nearest future survey within 7 days only.  
- Survey workbooks (`00_label_data/`); PHQ-9>=10 → `target_binary`. Text answers containing PHQ/GAD/DSM/Loneliness/target terms are blocklisted to prevent leakage.

**2) Feature engineering (multi-modal → tabular)**  
- Sensor: rolling mean/std/min/max/last (6–240h), ratios/deltas/z-scores/ranges, log1p (skewed), EWMs (6–72h), trends (12–168h), cross-wave deltas.  
- Text: time-of-day, low-cardinality one-hot, higher-cardinality ordinal + string length, with blocklist applied.  
- Voice: MFCC/deltas, spectral stats, chroma/tonnetz, pitch, voiced ratio, duration; aggregated per (ID, wave).  
- Outputs a single feature matrix with participant-level train/val splits (no cross-person leakage).

**3) Models (ML by default, DL optional)**  
- Tree-based ML: HistGradientBoosting, LightGBM, XGBoost (+ CatBoost optional).  
- Modalities: full multi-modal plus single-modality baselines (`HGB_sensor_only`, `Voice_HGB`, `Text_HGB`).  
- Fusion: late-fusion mean of modality holdouts; stacking (blend, logistic, XGB meta).  
- Optional DL: `--run-tabular-dl` trains MLP/TabNet baselines on the same tabular features.

**4) Training flow**  
- Optuna tuning per strategy (`--run-tuning`, default 50 trials) → saves `optuna_<strategy>_*.json`.  
- Rebuild tuned configs, evaluate on participant-stratified splits (CV with fallback for small N), collect holdout/CV AUC.  
- Stacking/late-fusion evaluated on top models. Best single model (fusion/stacking 제외) is rechecked on temporal block holdout (`--block-validation`) to show forward-looking performance.

**5) Outputs & where to find them**  
- Metrics: `results/<run>/full_pipeline_results_*.txt` (holdout/CV, stacking, late fusion, block validation).  
- Plots: `results/<run>/plots/` (ROC/PR/Calibration for best/stacking/fusion).  
- Tables: `logs/publication_table_metrics.(csv|md)` via `build_publication_tables.py`.  
- Feature importance: SHAP bar/beeswarm + TSV via `generate_feature_explanations.py`.  
- Model summaries: `plots/model_auc_summary.png`, `plots/model_block_vs_holdout.png` via `visualize_model_results.py`.  
- Data quality/coverage: timeline/coverage/missingness via `visualize_data_coverage.py`, `src/visualize_missing_data.py`, `visualize_interactive.py`.  
- PHQ-9 trajectories: `visualize_phq9_analysis.py` → `plots/phq9_timeline_interactive.html`.

**6) One-command run (recommended defaults)**  
```
python src/run_full_pipeline.py \
  --run-tuning \
  --history-hours 240 \
  --cv-folds 5 \
  --top-k-features 120 \
  --top-k-min 40 \
  --tuning-trials 50 \
  --use-gpu \
  --block-validation \
  --strategies hgb_top120 lgbm_full xgb_full HGB_sensor_only Voice_HGB Text_HGB \
  --output-dir results/test_run
```
Then:
```
python src/build_publication_tables.py --log-path results/test_run/full_pipeline_results_*.txt
python visualize_model_results.py --results-file results/test_run/full_pipeline_results_*.txt --output-dir plots
PYTHONPATH=src python src/generate_feature_explanations.py --optuna-json results/test_run/optuna_xgb_full_*.json --output-tsv results/test_run/shap_feature_importance.tsv --output-bar plots/shap_bar_top_features.svg --output-beeswarm plots/shap_beeswarm.svg
# Data quality/coverage
python visualize_data_coverage.py
python src/visualize_missing_data.py
python visualize_interactive.py
python visualize_phq9_analysis.py
```

## Figures/tables

- **Data coverage (timeline)**: `visualize_data_coverage.py` → `participant_timeline.png` (sensor spans + survey markers) and `sensor_coverage_detail.png` (per-sensor bars for selected participants). Good for “Data” section and missingness narrative.
- **PHQ-9 trajectories**: `visualize_phq9_analysis.py` → `phq9_timeline_interactive.html` (anonymized severity-colored markers, wave-to-wave change lines). Useful to show outcome distribution and change.
- **Interactive coverage**: `visualize_interactive.py` → `mindwatch_coverage_interactive.html` (anonymized hover, aggregated across all sensors). Use for supplementary materials/web appendix.
- **Missingness summary**: `src/visualize_missing_data.py` → `missing_raw_sensors.png`, `missing_tabular_features.png` to document completeness.
- **Model performance tables**: `build_publication_tables.py` on the latest `full_pipeline_results_*.txt` → `logs/publication_table_metrics.(csv|md)` for the Results section.
- **Model curves**: ROC/PR/Calibration plots emitted by `run_full_pipeline.py` (in `results/.../plots/`) and SHAP bar/beeswarm from `generate_feature_explanations.py`.
- **Model summary figures**: `visualize_model_results.py` → `plots/model_auc_summary.png` (holdout AUC bars incl. stacking/late fusion) and `plots/model_block_vs_holdout.png` (best model holdout vs. block validation).
## Data and leakage safeguards

- **Participant-level splits**: Train/val partitions are built by participant ID to avoid cross-person leakage (`StratifiedGroupKFold`, `participant_stratified_split`). Block-validation can be enabled (`--block-validation`) to test temporal robustness by holding out the most recent participants.
- **Sensor time shifting**: Daily aggregates (e.g., steps) are shifted by +1 day so values are only available after the day completes; hourly data are resampled to 1h and forward-filled with per-sensor medians.
- **Voice clips**: Each MP3 is aligned to the nearest *future* survey within 7 days; clips recorded after a survey are not used for that survey to prevent look-ahead. Voice features are cached in `logs/voice_features.csv` with versioned metadata.
- **Text features**: Survey free-text/categorical answers are encoded with one-hot/ordinal + length signals; PHQ/GAD/DSM/Loneliness and target fields are blocklisted to prevent label leakage. Caches live in `logs/text_features.csv` and track the configured label path.
- **Cross-wave controls**: Previous-wave deltas/prev_* features are added but can be dropped with `--exclude-prev-survey` for ablation or strict leakage avoidance.

## Feature engineering at a glance

- **Sensor (hourly)**: rolling mean/std/min/max/last for windows 6/12/24/48/72/240h; ratios and deltas vs rolling means; z-scores and ranges; log1p of skewed signals; exponential weighted means (spans 6–72h); linear trends over 12/24/72/168h; cross-wave deltas (current vs previous survey).
- **Static**: demographics (age, height, weight), categorical Sex encoded as codes.
- **Voice**: duration, RMS, ZCR, spectral centroid/bandwidth/rolloff/flatness/contrast, chroma, tonnetz, MFCC + deltas, pitch stats, voiced ratio; aggregated per (ID, wave) with clip_count.
- **Text**: time-of-day (bed/wake), low-cardinality one-hot, higher-cardinality ordinal encodings, string lengths.
- **Targets**: `target_binary` derived from PHQ-9>=10; regression target `target_score` retained for TFT.

## Modeling and evaluation

- **Strategy zoo**: Predefined HGB/LGBM/XGB/CatBoost configs (`build_default_strategies`) plus tuned variants from Optuna JSON. Voice-only (`Voice_HGB`), text-only (`Text_HGB`), and sensor-only branches included for ablation.
- **Tuning**: `src/tune_tabular_models.py` or the pipeline’s `--run-tuning` call Optuna with optional GPU, SQLite storage, and top-k feature sampling. Best params are cached as `optuna_<strategy>_*.json`.
- **Evaluation**: Holdout ROC-AUC plus grouped CV mean/std; optional block holdout; ROC/PR/Calibration plots saved to `plots/`. Stacking meta-models (mean blend, logistic, XGB) are reported but excluded from “best” single-model selection to keep block-validation stable.
- **Late fusion**: Sensor/voice/text means are reported when at least two modalities are available; recorded separately from single-model results.
- **Deep learning baselines**: `train_tabular_dl.py` trains MLP/TabNet on the engineered tabular features; TFT utilities live in `train_tft.py`.

## Caching and reproducibility

- Voice/text feature caches are versioned (metadata JSON) and invalidate automatically when source mtimes or feature schema versions change.
- Optuna studies can be persisted to SQLite via `--storage sqlite:///...` for resuming or cross-machine reproducibility.
- All key metrics, curves, and thresholds are written to `logs/`/`results/` with timestamps; `build_publication_tables.py` converts the latest run into CSV/Markdown tables for manuscripts.
- **HistGradientBoosting**: CPU-only.

## Troubleshooting

| Issue | Resolution |
| ----- | ---------- |
| `PerformanceWarning: DataFrame is highly fragmented` | Safe to ignore; emitted by pandas during feature construction. |
| Optuna tuning takes too long | Lower `--tuning-trials` or use `--n-jobs` for parallel CPU tuning. |
| Missing Optuna summaries in custom folders | The pipeline copies the latest `optuna_*.json` from `logs/` automatically; you can also provide `--optuna-dir`. |
| GPU warnings about fallback to CPU | Informational only; the run will continue using CPU methods. |

## Contributing

1. Create a branch.
2. Make changes and run the pipeline or targeted scripts to regenerate artifacts (kept locally).
3. Add updated plots/tables relevant to your change but keep large artifacts out of Git (the directories are ignored by default).
4. Submit a PR with a brief summary of the results (holdout metrics, tuning settings).

---

Questions or feedback? Open a private GitHub issue or reach the maintainers directly. Happy modeling! 🎯
