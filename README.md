# Mindwatch (Tabular AI for Mental Health)

Mindwatch converts multi-modal wearables, voice clips, and survey responses into publication-ready predictive models for depressive symptoms. The pipeline is built around leakage-safe feature engineering, automated hyperparameter tuning, and rich evaluation outputs (plots, tables, JSON summaries) that you can drop directly into papers.

## What's inside

- **Leakage-safe feature builder** (`src/run_tabular_models.py`)  
  Rolling statistics, cross-wave deltas, ratios, EWM trends, text encodings, and voice descriptors are created in one controlled place with participant-level splits to prevent data leakage.
- **Automation-ready pipeline** (`src/run_full_pipeline.py`)  
  Run tuning → evaluation → reporting with a single command; generates ROC/PR/calibration plots and publication tables automatically.
- **Optuna tuning CLI** (`src/tune_tabular_models.py`)  
  Supports CPU/GPU tuning, persistent SQLite studies, time-block validation, and sampled feature budgets.
- **Rich evaluation suite** (`src/evaluate_tabular_strategy.py`)  
  Outputs ROC/PR/calibration plots, confusion matrices, best-F1/F2 thresholds, and all metrics in JSON for reproducibility.
- **Tabular deep learning baselines** (`src/train_tabular_dl.py`)  
  MLP/TabNet training on the engineered feature set for comparison against tree-based methods.
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
 └─ train_tabular_dl.py          # MLP / TabNet baselines
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
