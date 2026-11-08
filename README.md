# SECOM Predictive Maintenance — Reproducible Benchmark

Leakage-controlled, uncertainty-aware benchmark of classifiers and imbalance
strategies for rare-failure detection on the UCI **SECOM** dataset
(1,567 samples, **590** features, 6.6% minority class).

This README documents how to reproduce the paper's results and records the
honest, re-run findings. It supports the manuscript *"Robust Predictive
Maintenance on the SECOM Dataset: A Leakage-Controlled, Uncertainty-Aware
Benchmark under Extreme Class Imbalance"* (Regmi, Khatiwada, Shakya).

## 1. Data

SECOM is not stored in this repo (see `.gitignore`). Download from UCI and
place both files in a `secom/` directory:

- `secom/secom.data`   — 1,567 × 590 space-separated sensor values
- `secom/secom_labels.data` — label (−1 pass / 1 fail) + timestamp per row

UCI: https://archive.ics.uci.edu/dataset/179/secom

## 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Versions are **pinned** (`requirements.txt`). This matters: scikit-learn
changed its calibration API (`cv="prefit"` was removed in ≥1.6), which silently
disabled calibration in the original code. Re-pin and re-run if you change the
stack, and regenerate every reported number so the paper and code stay in sync.

## 3. Run

```bash
python main.py            # full pipeline, writes to output/
```

Set `run_multiseed = True` in `main.py` to produce the 10-seed mean ± std
table (this is the primary basis for all performance claims and must be run).

## 4. Reproducible results (10 seeds, audit re-run)

Performance is **unstable on this extreme-imbalance task**; single-split point
metrics overstate achievable performance. Leading configurations
(mean ± std F1 over 10 seeds):

| Model + sampler | F1 | ROC-AUC | Threshold |
|---|---|---|---|
| Balanced Bagging + cost-sensitive | 0.223 ± 0.079 | 0.744 | 0.510 ± 0.112 |
| Extra Trees + SMOTE | 0.215 ± 0.031 | 0.714 | 0.295 ± 0.055 |
| Balanced Bagging + Borderline-SMOTE | 0.192 ± 0.058 | 0.686 | 0.211 ± 0.124 |
| Gradient Boosting + Borderline-SMOTE | 0.190 ± 0.115 | 0.720 | 0.157 ± 0.138 |
| Balanced Bagging + SMOTE | 0.188 ± 0.059 | 0.692 | 0.255 ± 0.115 |
| Gradient Boosting + SVM-SMOTE | 0.137 ± 0.127 | 0.730 | 0.217 ± 0.225 |

Key findings:

- **No configuration is statistically dominant.** McNemar on the two leading
  models (Balanced Bagging + SMOTE vs. Extra Trees + SMOTE) gives p = 0.43.
  Cost-sensitive learning is competitive with oversampling.
- **Calibration** (isotonic, corrected API): Brier 0.068 → 0.057, ECE
  0.093 → 0.020; mean predicted probability 0.065 ≈ true failure rate 0.067.
- **Distribution shift:** a few features differ at raw p < 0.05, **none** after
  Bonferroni correction. (This is a train-vs-test check, not concept drift.)
- **Temporal split (chronological):** discrimination collapses to chance
  (ROC-AUC ≈ 0.46), i.e. random-split results do **not** transfer to realistic
  deployment. Production use requires retraining and drift monitoring.

## 5. Fixes applied (vs. the original code)

| File | Fix | Why |
|---|---|---|
| `models/calibration.py` | Use `FrozenEstimator` for isotonic calibration; warn instead of silently returning an uncalibrated model | `cv="prefit"` removed in sklearn ≥1.6 silently disabled calibration |
| `models/tuning.py` | Resampler nested **inside** the GridSearchCV pipeline | Original resampled before CV, leaking SMOTE points across folds (CV scores inflated to ~1.0) |
| `models/registry.py` | Removed AdaBoost `algorithm="SAMME"` kwarg | Removed in sklearn ≥1.6 (raised `TypeError`) |

## 6. Reproducibility notes

- All preprocessing, resampling, calibration and threshold selection are fit on
  training/calibration data only; the test set is untouched.
- Fixed `random_state` (default 42); multi-seed runner sweeps seeds 0–9.
- Exact legacy point values from the previous revision (e.g. F1 = 0.340,
  threshold 0.06) were produced by an earlier code version and are **not**
  reproducible on this pinned stack — report the multi-seed results instead.
