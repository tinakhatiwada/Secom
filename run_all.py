"""
run_all.py — Reproduce every reported result for the SECOM benchmark.

Run after applying the audit fixes to AdvancedSECOM/ (calibration.py, tuning.py,
registry.py) and placing the data in ./secom/. Outputs go to ./output/ so the
repository itself is the evidence behind the paper's tables and figures.

Produces:
  output/table1_multiseed.csv   - 10-seed mean±std for leading configs (Table 1)
  output/mcnemar.csv            - pairwise McNemar p-values (answers R1-Q4)
  output/calibration.json       - Brier / ECE, uncalibrated vs isotonic
  output/reliability.png        - reliability diagram (Figure 3)
  output/chronological.json     - temporal-split result (§5.6)

Usage:  python run_all.py            (full; ~15 min)
        python run_all.py --quick    (3 seeds, for a smoke test)
"""
from __future__ import annotations
import argparse, json, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from AdvancedSECOM import AdvancedSECOM
from sklearn.base import clone
from sklearn.metrics import (f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix)
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from statsmodels.stats.contingency_tables import mcnemar

DATA = Path("secom")
OUT  = Path("output"); OUT.mkdir(exist_ok=True)
SEED = 42

# Leading configurations reported in Table 1.
def models(rs):
    return {
        "Balanced Bagging": BalancedBaggingClassifier(n_estimators=100, random_state=rs),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, max_depth=15,
                                            random_state=rs, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                            max_depth=8, random_state=rs, subsample=0.8,
                                            min_samples_leaf=5),
    }
def samplers(rs):
    return {"Cost-sensitive": None,
            "SMOTE": SMOTE(random_state=rs, k_neighbors=3),
            "Borderline-SMOTE": BorderlineSMOTE(random_state=rs, kind="borderline-1"),
            "SVM-SMOTE": SVMSMOTE(random_state=rs)}
CONFIGS = [("Balanced Bagging","Cost-sensitive"),("Extra Trees","SMOTE"),
           ("Balanced Bagging","Borderline-SMOTE"),("Gradient Boosting","Borderline-SMOTE"),
           ("Balanced Bagging","SMOTE"),("Gradient Boosting","SVM-SMOTE")]

def prep(seed):
    s = AdvancedSECOM(data_dir=DATA, output_dir=OUT/f"_seed{seed}", random_state=seed)
    assert s.load_data(), "Place secom.data and secom_labels.data in ./secom/"
    s.split_data(random_state=seed)
    s.advanced_preprocessing(use_knn_imputer=False, use_robust_scaler=True,
                             feature_selection_method="rfe")
    return s

def evaluate(s, model, sampler):
    if sampler is None: Xr, yr = s.X_train.copy(), s.y_train.copy()
    else: Xr, yr = clone(sampler).fit_resample(s.X_train, s.y_train)
    m = clone(model).fit(Xr, yr)
    pm, thr = s._calibrate_and_threshold(m, s.X_cal, s.y_cal)
    proba = pm.predict_proba(s.X_test)[:, 1]
    pred = (proba >= thr).astype(int)
    return pred, proba, float(thr)

def metrics(y, pred, proba):
    return dict(F1=f1_score(y, pred, zero_division=0), MCC=matthews_corrcoef(y, pred),
                ROC_AUC=roc_auc_score(y, proba), PR_AUC=average_precision_score(y, proba),
                recall=recall_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0))

def table1(seeds):
    rows = []
    for seed in seeds:
        s = prep(seed); M, S = models(seed), samplers(seed); yt = s.y_test.values
        for mn, sn in CONFIGS:
            pred, proba, thr = evaluate(s, M[mn], S[sn])
            if int(pred.sum()) < 2: continue
            r = metrics(yt, pred, proba); r.update(seed=seed, config=f"{mn} + {sn}", threshold=thr)
            rows.append(r)
    df = pd.DataFrame(rows)
    agg = df.groupby("config").agg(
        F1_mean=("F1","mean"), F1_std=("F1","std"), MCC_mean=("MCC","mean"),
        PRAUC_mean=("PR_AUC","mean"), ROC_mean=("ROC_AUC","mean"),
        recall_mean=("recall","mean"), precision_mean=("precision","mean"),
        thr_mean=("threshold","mean"), thr_std=("threshold","std"),
        n_seeds=("seed","nunique")).sort_values("F1_mean", ascending=False).round(4)
    agg.to_csv(OUT/"table1_multiseed.csv")
    df.to_csv(OUT/"per_seed_metrics.csv", index=False)   # per-seed values for paired tests
    print("\n[Table 1] multi-seed:\n", agg.to_string()); return agg

def significance(headline="Balanced Bagging + SMOTE"):
    """Proper test for 'is the headline model better?': paired Wilcoxon / t-test
    on the per-seed F1 values (not single-seed McNemar), plus a bootstrap CI of
    the mean F1 difference. Answers R1-Q4 at the seed level."""
    from scipy import stats
    df = pd.read_csv(OUT/"per_seed_metrics.csv")
    piv = df.pivot(index="seed", columns="config", values="F1").dropna()
    a = piv[headline].values
    rows = []
    for c in piv.columns:
        if c == headline: continue
        b = piv[c].values
        diff = a - b
        boot = [np.mean(np.random.choice(diff, len(diff), replace=True)) for _ in range(5000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(vs=c, mean_diff=round(float(diff.mean()), 4),
                         ci95=f"[{lo:+.3f}, {hi:+.3f}]",
                         paired_t_p=round(float(stats.ttest_rel(a, b).pvalue), 3),
                         wilcoxon_p=round(float(stats.wilcoxon(a, b).pvalue), 3)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT/"significance_tests.csv", index=False)
    print(f"\n[Significance] paired tests of {headline} vs each config (per-seed F1):\n",
          out.to_string(index=False))
    return out

def mcnemar_table():
    s = prep(SEED); yt = s.y_test.values; M, S = models(SEED), samplers(SEED)
    preds = {f"{mn}+{sn}": evaluate(s, M[mn], S[sn])[0] for mn, sn in CONFIGS}
    rows = []
    for a, b in itertools.combinations(preds, 2):
        ca, cb = preds[a] == yt, preds[b] == yt
        n10, n01 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
        tab = np.array([[int(np.sum(ca & cb)), n10], [n01, int(np.sum(~ca & ~cb))]])
        p = mcnemar(tab, exact=(n10 + n01) < 25).pvalue
        rows.append(dict(A=a, B=b, A_only=n10, B_only=n01, p_value=round(float(p), 4)))
    df = pd.DataFrame(rows)
    # Bonferroni correction over all pairwise comparisons (consistent with the
    # drift analysis). Single-seed McNemar is secondary to the paired test above.
    k = len(df); alpha_bonf = 0.05 / k
    df["sig_raw_0.05"] = df["p_value"] < 0.05
    df["sig_bonferroni"] = df["p_value"] < alpha_bonf
    df.to_csv(OUT/"mcnemar.csv", index=False)
    print(f"\n[McNemar] seed {SEED} ({k} comparisons; Bonferroni alpha={alpha_bonf:.4f}; "
          f"raw sig={int(df['sig_raw_0.05'].sum())}, Bonferroni sig={int(df['sig_bonferroni'].sum())}):\n",
          df.to_string(index=False))
    return df

def calibration():
    s = prep(SEED); yt = s.y_test.values
    Xr, yr = clone(samplers(SEED)["SMOTE"]).fit_resample(s.X_train, s.y_train)
    base = clone(models(SEED)["Balanced Bagging"]).fit(Xr, yr)
    p_raw = base.predict_proba(s.X_test)[:, 1]
    pm, _ = s._calibrate_and_threshold(base, s.X_cal, s.y_cal)
    p_cal = pm.predict_proba(s.X_test)[:, 1]
    def ece(p, b=10):
        e = np.linspace(0, 1, b + 1); t = 0.0
        for i in range(b):
            m = (p >= e[i]) & ((p < e[i + 1]) if i < b - 1 else (p <= e[i + 1]))
            if m.sum(): t += m.mean() * abs(p[m].mean() - yt[m].mean())
        return float(t)
    res = dict(brier_uncal=float(brier_score_loss(yt, p_raw)),
               brier_isotonic=float(brier_score_loss(yt, p_cal)),
               ece_uncal=ece(p_raw), ece_isotonic=ece(p_cal),
               base_rate=float(yt.mean()), mean_uncal=float(p_raw.mean()),
               mean_isotonic=float(p_cal.mean()))
    (OUT/"calibration.json").write_text(json.dumps(res, indent=2))
    fig, ax = plt.subplots(figsize=(6.2, 6))
    for p, lab in [(p_raw, "Uncalibrated"), (p_cal, "Isotonic")]:
        ft, mp = calibration_curve(yt, p, n_bins=8, strategy="quantile")
        ax.plot(mp, ft, "o-", label=f"{lab} (Brier={brier_score_loss(yt,p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.axhline(float(yt.mean()), color="gray", ls=":", label=f"Base rate={yt.mean():.3f}")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed failure frequency",
           xlim=(0, 0.6), ylim=(0, 0.6), title="Reliability — Balanced Bagging + SMOTE")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(OUT/"reliability.png", dpi=150)
    print("\n[Calibration]:\n", json.dumps(res, indent=2)); return res

def chronological():
    raw = (DATA/"secom_labels.data").read_text().strip().splitlines()
    import datetime as dt
    ts, y = [], []
    for line in raw:
        parts = line.replace('"', '').split()
        y.append(0 if int(float(parts[0])) == -1 else 1)
        try: ts.append(dt.datetime.strptime(" ".join(parts[1:3]), "%d/%m/%Y %H:%M:%S"))
        except Exception: ts.append(dt.datetime.max)
    X = pd.read_csv(DATA/"secom.data", sep=r"\s+", header=None).values
    order = np.argsort(ts); X, y = X[order], np.array(y)[order]
    n = len(X); n_te = int(n*0.2); n_cal = int((n-n_te)*0.2)
    Xtr, ytr = X[:n-n_te-n_cal], y[:n-n_te-n_cal]
    Xcal, ycal = X[n-n_te-n_cal:n-n_te], y[n-n_te-n_cal:n-n_te]
    Xte, yte = X[n-n_te:], y[n-n_te:]
    imp = SimpleImputer(strategy="median"); sc = RobustScaler()
    Xtr = sc.fit_transform(imp.fit_transform(Xtr)); Xcal = sc.transform(imp.transform(Xcal)); Xte = sc.transform(imp.transform(Xte))
    sel = RFE(RandomForestClassifier(n_estimators=50, random_state=SEED),
              n_features_to_select=min(200, Xtr.shape[1]))
    Xtr = sel.fit_transform(Xtr, ytr); Xcal = sel.transform(Xcal); Xte = sel.transform(Xte)
    sh = AdvancedSECOM(data_dir=DATA, output_dir=OUT/"_chrono", random_state=SEED)
    sh.X_cal, sh.y_cal = pd.DataFrame(Xcal), pd.Series(ycal)
    Xr, yr = SMOTE(random_state=SEED, k_neighbors=3).fit_resample(pd.DataFrame(Xtr), pd.Series(ytr))
    base = BalancedBaggingClassifier(n_estimators=100, random_state=SEED).fit(Xr, yr)
    pm, thr = sh._calibrate_and_threshold(base, sh.X_cal, sh.y_cal)
    proba = pm.predict_proba(pd.DataFrame(Xte))[:, 1]; pred = (proba >= thr).astype(int)
    res = dict(split="chronological", **metrics(yte, pred, proba), threshold=float(thr),
               test=len(yte), test_pos=int(yte.sum()))
    (OUT/"chronological.json").write_text(json.dumps(res, default=float, indent=2))
    print("\n[Chronological split]:\n", json.dumps(res, default=float, indent=2)); return res

def reproduce_all(data_dir=None, out_dir=None, seeds=None):
    """Single entry point for all reviewer-facing evidence. Callable from
    main.py so there is ONE multi-seed path, not two.
    data_dir : folder holding secom.data + secom_labels.data (default ./secom/)
    out_dir  : where CSV/JSON/PNG go (default ./output/)
    seeds    : seed list (default = seed 42 plus nine further seeds, 42,1..9)
    """
    global DATA, OUT
    if data_dir is not None:
        DATA = Path(data_dir)
    if out_dir is not None:
        OUT = Path(out_dir); OUT.mkdir(parents=True, exist_ok=True)
    if seeds is None:
        seeds = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print("=" * 60, "\nReproducing SECOM benchmark results (seeds:",
          list(seeds), ")\n", "=" * 60)
    table1(seeds); significance(); mcnemar_table(); calibration(); chronological()
    print(f"\nAll outputs written to {OUT}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 seeds instead of 10")
    args = ap.parse_args()
    # Seed 42 (the main-experiment seed) plus further seeds, matching the
    # manuscript text and revision_experiments.py. NOT range(10) (0-9).
    reproduce_all(seeds=[42, 1, 2] if args.quick else [42, 1, 2, 3, 4, 5, 6, 7, 8, 9])
