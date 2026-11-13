"""Expanding-window chronological evaluation.
Train on first f%, test on the next 10% block, for f in {60,70,80,90}%.
Windows-ready: set DATA_DIR below to the folder that CONTAINS the two files
  secom.data  and  secom_labels.data
"""
import warnings, json, datetime as dt, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (f1_score, matthews_corrcoef, roc_auc_score,
    average_precision_score, recall_score, precision_score)
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE

# ---- EDIT THIS ONE LINE: folder holding secom.data and secom_labels.data ----
DATA_DIR = Path(r"C:\Users\sansk\Downloads\Secom\secom")
SEED = 42

# locate the files (tries DATA_DIR, then a 'secom' or 'data/secom' subfolder)
def _find():
    for d in [DATA_DIR, DATA_DIR/"secom", DATA_DIR/"data"/"secom"]:
        if (d/"secom_labels.data").exists() and (d/"secom.data").exists():
            return d
    raise FileNotFoundError(
        f"Could not find secom.data AND secom_labels.data under {DATA_DIR}. "
        f"Set DATA_DIR to the folder that directly contains both files.")
DATA = _find()
print("Using data folder:", DATA)

raw = (DATA/"secom_labels.data").read_text().strip().splitlines()
ts, y = [], []
for line in raw:
    p = line.replace('"', '').split()
    y.append(0 if int(float(p[0])) == -1 else 1)
    ts.append(dt.datetime.strptime(" ".join(p[1:3]), "%d/%m/%Y %H:%M:%S"))
X = pd.read_csv(DATA/"secom.data", sep=r"\s+", header=None).values
order = np.argsort(ts); X = X[order]; y = np.array(y)[order]; n = len(X)

def opt_threshold(yv, pv):
    best = (0, 0.5)
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(yv, (pv >= t).astype(int), zero_division=0)
        if f > best[0]: best = (f, round(t, 2))
    return best[1]

rows = []
for ftrain in [0.6, 0.7, 0.8, 0.9]:
    tr_end = int(n*ftrain); te_end = int(n*(ftrain+0.1))
    cal_start = int(tr_end*0.8)                 # calibration = last 20% of the train period (still in the past)
    Xtr, ytr = X[:cal_start], y[:cal_start]
    Xcal, ycal = X[cal_start:tr_end], y[cal_start:tr_end]
    Xte, yte = X[tr_end:te_end], y[tr_end:te_end]
    if yte.sum() < 2 or ytr.sum() < 10:
        rows.append(dict(train_pct=int(ftrain*100), skipped=True)); continue
    imp = SimpleImputer(strategy="median"); sc = RobustScaler()
    Xtr = sc.fit_transform(imp.fit_transform(Xtr)); Xcal = sc.transform(imp.transform(Xcal)); Xte = sc.transform(imp.transform(Xte))
    sel = RFE(RandomForestClassifier(n_estimators=50, random_state=SEED), n_features_to_select=min(200, Xtr.shape[1]))
    Xtr = sel.fit_transform(Xtr, ytr); Xcal = sel.transform(Xcal); Xte = sel.transform(Xte)
    Xr, yr = SMOTE(random_state=SEED, k_neighbors=min(3, int(ytr.sum())-1)).fit_resample(Xtr, ytr)
    base = BalancedBaggingClassifier(n_estimators=100, random_state=SEED).fit(Xr, yr)
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(Xcal, ycal)
    thr = opt_threshold(ycal, cal.predict_proba(Xcal)[:, 1])
    proba = cal.predict_proba(Xte)[:, 1]; pred = (proba >= thr).astype(int)
    roc = roc_auc_score(yte, proba) if len(set(yte)) > 1 else float("nan")
    p = yte.mean(); flag_all_f1 = 2*p/(1+p)
    rows.append(dict(train_pct=int(ftrain*100), test_n=len(yte), test_fail=int(yte.sum()),
        F1=round(f1_score(yte, pred, zero_division=0), 3), MCC=round(matthews_corrcoef(yte, pred), 3),
        ROC_AUC=round(roc, 3), PR_AUC=round(average_precision_score(yte, proba), 3),
        recall=round(recall_score(yte, pred, zero_division=0), 3),
        precision=round(precision_score(yte, pred, zero_division=0), 3),
        flag_all_F1=round(flag_all_f1, 3), threshold=thr))

df = pd.DataFrame(rows)
print(df.to_string(index=False))
valid = df[~df.get("skipped", pd.Series([False]*len(df))).fillna(False)]
if len(valid):
    print("\nMean ROC-AUC: %.3f  |  Mean F1: %.3f  |  random-split ROC-AUC for reference: ~0.69"
          % (valid.ROC_AUC.mean(), valid.F1.mean()))
(DATA.parent/"expanding_window.json").write_text(df.to_json(orient="records"))
print("\nSaved:", DATA.parent/"expanding_window.json")
