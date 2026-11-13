"""
Nested cross-validation for Balanced Bagging + SMOTE on SECOM (Windows-ready).

Rebuilds the FULL final pipeline inside every outer fold, so the result is
directly comparable to the single test-split F1:
  per-fold median impute -> correlation filter -> RobustScaler -> RFE
  -> fold-internal train/calibration split
  -> SMOTE on fold-train -> BalancedBagging
  -> isotonic calibration on fold-cal -> F1-optimal threshold on fold-cal
  -> score on the outer held-out fold

Prints two numbers on the SAME folds:
  OLD protocol  : uncalibrated model, default 0.50 cutoff   (the apples-to-apples
                  version of the paper's CV-F1; drops once RFE is fitted per fold)
  NESTED/fixed  : calibrated + tuned threshold               (comparable to test F1)

EDIT ONE LINE: set DATA_DIR to the folder holding secom.data and secom_labels.data.
"""
import warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator            # scikit-learn >= 1.6
from sklearn.metrics import f1_score
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE

# ---- EDIT THIS ONE LINE: folder holding secom.data and secom_labels.data ----
DATA_DIR = Path(r"C:\Users\sansk\Downloads\Secom\secom")
SEED = 42
N_SPLITS = 5          # outer folds
RFE_STEP = 25         # features removed per RFE round (1 = slow/faithful, 25 = fast)

def _find():
    for d in [DATA_DIR, DATA_DIR/"secom", DATA_DIR/"data"/"secom"]:
        if (d/"secom_labels.data").exists() and (d/"secom.data").exists():
            return d
    raise FileNotFoundError(
        f"Could not find secom.data AND secom_labels.data under {DATA_DIR}. "
        f"Set DATA_DIR to the folder that directly contains both files.")
DATA = _find(); print("Using data folder:", DATA)

X = pd.read_csv(DATA/"secom.data", sep=r"\s+", header=None).values
y = np.array([0 if int(float(l.split()[0])) == -1 else 1
              for l in (DATA/"secom_labels.data").read_text().strip().splitlines()])
print(f"N={len(X)}  failures={y.sum()} ({100*y.mean():.2f}%)\n")

def opt_threshold(yv, pv):
    best = (0.0, 0.5)
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(yv, (pv >= t).astype(int), zero_division=0)
        if f > best[0]: best = (f, round(float(t), 2))
    return best[1]

def preprocess_fit(Xtr, ytr):
    imp = SimpleImputer(strategy="median"); Xtr = imp.fit_transform(Xtr)
    df = pd.DataFrame(Xtr); cor = df.corr().abs()
    upper = cor.where(np.triu(np.ones(cor.shape), 1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > 0.95).any()]
    keep = [c for c in df.columns if c not in drop]
    Xtr = df[keep].values
    sc = RobustScaler(); Xtr = sc.fit_transform(Xtr)
    nsel = min(200, Xtr.shape[1])
    sel = RFE(RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1),
              n_features_to_select=nsel, step=RFE_STEP).fit(Xtr, ytr)
    return (imp, keep, sc, sel), sel.transform(Xtr)

def preprocess_apply(pp, Xv):
    imp, keep, sc, sel = pp
    Xv = pd.DataFrame(imp.transform(Xv))[keep].values
    return sel.transform(sc.transform(Xv))

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
old_f1, new_f1, thrs = [], [], []
print(f"Running {N_SPLITS}-fold nested CV (each fold prints as it finishes)...")
for k, (tr, va) in enumerate(skf.split(X, y), 1):
    print(f"  fold {k} fitting...", flush=True)
    Xtr, ytr = X[tr], y[tr]; Xva, yva = X[va], y[va]
    pp, Xtr_p = preprocess_fit(Xtr, ytr); Xva_p = preprocess_apply(pp, Xva)
    Xf, Xc, yf, yc = train_test_split(Xtr_p, ytr, test_size=0.20,
                                      random_state=SEED, stratify=ytr)
    Xr, yr = SMOTE(random_state=SEED, k_neighbors=min(5, int(yf.sum())-1)).fit_resample(Xf, yf)
    base = BalancedBaggingClassifier(n_estimators=100, random_state=SEED).fit(Xr, yr)
    p_old = base.predict_proba(Xva_p)[:, 1]
    old_f1.append(f1_score(yva, (p_old >= 0.50).astype(int), zero_division=0))
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(Xc, yc)
    thr = opt_threshold(yc, cal.predict_proba(Xc)[:, 1]); thrs.append(thr)
    p_new = cal.predict_proba(Xva_p)[:, 1]
    new_f1.append(f1_score(yva, (p_new >= thr).astype(int), zero_division=0))
    print(f"    OLD(0.50)={old_f1[-1]:.3f}   NESTED(thr={thr:.2f})={new_f1[-1]:.3f}", flush=True)

old_f1, new_f1 = np.array(old_f1), np.array(new_f1)
print(f"\nOLD protocol  (uncalibrated, 0.50): F1 = {old_f1.mean():.3f} +/- {old_f1.std():.3f}")
print(f"NESTED/fixed  (calibrated, tuned) : F1 = {new_f1.mean():.3f} +/- {new_f1.std():.3f}")
print("  (compare: single-split test F1 = 0.340; ten-seed test F1 = 0.203)")
Path(DATA.parent/"nested_cv.json").write_text(json.dumps(dict(
    old_mean=float(old_f1.mean()), old_std=float(old_f1.std()),
    nested_mean=float(new_f1.mean()), nested_std=float(new_f1.std()),
    thresholds=thrs, old_folds=old_f1.tolist(), nested_folds=new_f1.tolist()), indent=2))
print("saved", DATA.parent/"nested_cv.json")
