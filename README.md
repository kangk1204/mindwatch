# Mindwatch (Tabular AI for Mental Health)

Mindwatch is an end-to-end tabular modeling pipeline that predicts depressive symptoms from multi-modal wearable/EMA data. It includes feature engineering, baseline model sweeps, hyperparameter tuning with Optuna, and rich evaluation artifacts (ROC curves, MCC/F1/F2 tables) suitable for publications.

## Highlights

- **Leakage-safe feature builder**: Adds rolling stats, ratios, deltas, exponential moving averages, cross-wave survey deltas, etc. in a single place (`run_tabular_models.py`).
- **Strategy factory**: A reusable set of configurations (HGB, LightGBM, XGBoost, CatBoost) plus the latest Optuna-tuned `HGB_optuna_best`.
- **Optuna CLI (`tune_tabular_models.py`)**: Supports CPU/GPU, persistent studies, and optional block validation.
- **Evaluation suite (`evaluate_tabular_strategy.py`)**: Generates ROC plots, F1/F2/MCC threshold tables, and JSON summaries for paper-ready figures.

## Repository Layout

```
00_input_data/          # hourly sensor data (not tracked; user-provided)
00_label_data/          # survey/label sheets (not tracked)
logs/                   # Optuna + evaluation outputs (created automatically)
plots/                  # ROC curve images (created automatically)
src/
 ├─ run_tabular_models.py       # feature building + strategy runner
 ├─ tune_tabular_models.py      # Optuna tuning CLI
 ├─ evaluate_tabular_strategy.py# ROC/MCC/F1/F2 evaluation CLI
 └─ train_tft.py                # shared data-loading utilities
```

## Requirements

- Python 3.10+ (tested on 3.12)
- OS: Ubuntu 22.04 LTS / macOS 13+ (Apple Silicon supported)
- Packages: `pandas`, `numpy`, `scikit-learn`, `optuna`, `lightgbm`, `xgboost`, `catboost`, `matplotlib`, `torch`, etc.  
  *(See `requirements.txt` if provided, otherwise install manually as below.)*
- GPU tuning (optional): CUDA-compatible device for Linux; Apple Silicon uses CPU (Metal-accelerated CatBoost is experimental).

> **Note**: raw data files are not included. Place your sensor CSVs under `00_input_data/` and survey sheets in `00_label_data/`, following the naming expected by `train_tft.py`.

## Installation (Ubuntu & macOS)

```bash
# Clone your private repo
git clone git@github.com:<YOUR_USERNAME>/mindwatch.git
cd mindwatch

# (Ubuntu) system deps
sudo apt-get update && sudo apt-get install -y build-essential python3-venv

# (macOS) ensure Homebrew + Command Line Tools are installed if not already
xcode-select --install  # once

# Create virtualenv (works on both Ubuntu and Mac)
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip & install runtime deps
pip install --upgrade pip
pip install pandas numpy scikit-learn optuna lightgbm xgboost catboost matplotlib torch pytorch-forcasting
```

If you have a `requirements.txt`, simply run `pip install -r requirements.txt`.

## Quickstart

### Ubuntu
```bash
source .venv/bin/activate
# Baseline sweep of built-in strategies
python src/run_tabular_models.py --history-hours 240 --cv-folds 5

# Optuna tuning (CPU)
python src/tune_tabular_models.py \
  --strategy hgb_top120 \
  --history-hours 240 \
  --cv-folds 5 \
  --trials 50 \
  --block-validation

# Rich evaluation for best strategy
python src/evaluate_tabular_strategy.py \
  --strategy HGB_optuna_best \
  --history-hours 240 \
  --cv-folds 5
```

### macOS (Apple Silicon / Intel)
macOS steps are identical. When creating the virtualenv, ensure you use the system Python 3.10+ or install via `pyenv`/Homebrew.  
GPU acceleration is limited on Apple Silicon; run with `--use-gpu` **off**.

## Commands in Detail

### Baseline sweep
Runs all predefined strategies, logs to `logs/tabular_results.txt`, and performs block validation on the best holdout.
```bash
python src/run_tabular_models.py --history-hours 240 --cv-folds 5
```

### Optuna tuning
Tune a specific model family. Example (HGB, GPU-enabled on Linux, resume-capable via SQLite):
```bash
python src/tune_tabular_models.py \
  --strategy hgb_top120 \
  --history-hours 240 \
  --cv-folds 5 \
  --trials 200 \
  --split-seed 42 \
  --block-validation \
  --use-gpu \
  --study-name tabular_hgb \
  --storage sqlite:///logs/optuna_hgb.db
```
Outputs are written to `logs/optuna_<strategy>_<timestamp>.json` plus the SQLite study.

### Rich evaluation (ROC/F1/MCC)
```bash
python src/evaluate_tabular_strategy.py \
  --strategy HGB_optuna_best \
  --history-hours 240 \
  --cv-folds 5 \
  --split-seed 42
```
Produces:
- `logs/evaluation_<strategy>_<timestamp>.json`
- `plots/roc_<strategy>_<timestamp>.png`

The JSON includes ROC-AUC, ROC curve samples, best-F1/F2 thresholds, MCC, accuracy, specificity, etc.

## GPU Notes
- **XGBoost**: `--use-gpu` sets `tree_method="gpu_hist"` and `predictor="gpu_predictor"`.
- **LightGBM**: uses `device="gpu"` when available (ensure LightGBM built with GPU support).
- **CatBoost**: currently runs on CPU; add `task_type="GPU"` manually if needed.
- **HistGradientBoosting** (HGB) is CPU-only.

## Troubleshooting

| Issue | Fix |
| ----- | --- |
| `PerformanceWarning: DataFrame is highly fragmented` | Harmless; pandas warns when many columns are appended. Can be ignored or refactored later. |
| `ModuleNotFoundError` | Ensure virtualenv is active and dependencies installed. |
| Optuna command runs slow | Consider reducing `--trials`, or use `--n-jobs` for parallel CPU trials. |
| Need to resume interrupted tuning | Use the same `--study-name` + `--storage sqlite:///...` combination; Optuna resumes automatically. |

## Contributing

1. Create a new branch.
2. Make changes and update tests/plots if relevant.
3. Submit PR; include latest ROC/evaluation artifacts for reproducibility.

---

Questions or issues? Open a private GitHub issue in the `mindwatch` repo or contact the maintainers. Happy modeling! 🎯
### Tabular deep learning (MLP / TabNet)
Requires PyTorch (already used by TFT) and optionally `pytorch-tabnet` (`pip install pytorch-tabnet`).
```bash
python src/train_tabular_dl.py \
  --model-type mlp \
  --history-hours 240 \
  --epochs 50 \
  --batch-size 256 \
  --exclude-prev-survey  # optional ablation
```
Use `--model-type tabnet` to switch architectures; `--use-gpu` is not needed because Torch auto-detects CUDA. Results are saved to `logs/tabular_<model>_<scenario>_<timestamp>.json`.
