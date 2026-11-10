
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import time
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, IsolationForest,
    RandomForestClassifier, StackingClassifier, VotingClassifier,
)
from sklearn.feature_selection import (
    RFE, SelectFromModel, SelectKBest, f_classif,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold,
    cross_val_score, learning_curve, train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ── scipy ─────────────────────────────────────────────────────────────────────
try:
    from scipy.stats import chi2 as _chi2, ks_2samp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ── imbalanced-learn ──────────────────────────────────────────────────────────
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

# ── optional heavy deps ───────────────────────────────────────────────────────
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available.  pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available.  pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available.  pip install catboost")

try:
    import shap
    _ = shap.__version__
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    print("SHAP not available.  pip install shap")

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available.  pip install lime")

try:
    from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("statsmodels not available.  pip install statsmodels")


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper
# ═══════════════════════════════════════════════════════════════════════════════

def fmt(v, spec=".4f", fallback="N/A"):
    """Format a numeric value, or return fallback if None."""
    return format(v, spec) if v is not None else fallback


# ═══════════════════════════════════════════════════════════════════════════════
#  AdvancedSECOM — all-in-one class
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedSECOM:
    """Publication-ready SECOM predictive maintenance pipeline (standalone)."""

    def __init__(self, data_dir=None, output_dir=None,
                 cost_fp: int = 100, cost_fn: int = 10_000,
                 random_state: int = 42):
        self.X = self.y = None
        self.X_train = self.X_test = self.X_cal = None
        self.y_train = self.y_test = self.y_cal = None
        self.results: dict = {}
        self.feature_names: list = []
        self.class_weight_ratio: float = 1.0
        self.random_state = random_state
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.data_dir = Path(data_dir) if data_dir else Path.cwd() / "secom"
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fmt(v, spec=".4f", fallback="N/A"):
        return fmt(v, spec, fallback)

    # ── Data Loading ──────────────────────────────────────────────────────────

    def load_data(self) -> bool:
        try:
            print("Loading SECOM dataset …")
            data = pd.read_csv(self.data_dir / "secom.data",
                               sep=r"\s+", header=None)
            labels = pd.read_csv(self.data_dir / "secom_labels.data",
                                 sep=r"\s+", header=None)
            data["label"] = labels.iloc[:, 0]
            self.X = data.drop("label", axis=1)
            self.y = data["label"]
            print(f"Dataset shape : {data.shape}")
            print(f"Label dist    :\n{self.y.value_counts()}")
            if len(self.X) < 2000:
                print("⚠  Small dataset (<2 000). Use repeated k-fold.")
            self.y = self.y.map({-1: 0, 1: 1})
            cnt = self.y.value_counts()
            self.class_weight_ratio = cnt[0] / max(cnt.get(1, 1), 1)
            print(f"Imbalance : {self.class_weight_ratio:.2f}:1 | "
                  f"Minority: {cnt.get(1,0)/len(self.y)*100:.2f}%")
        except Exception as e:
            print(f"Load error: {e}"); traceback.print_exc(); return False
        return True

    # ── Train / Cal / Test Split ──────────────────────────────────────────────

    def split_data(self, random_state=None):
        rs = self.random_state if random_state is None else random_state
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=rs, stratify=self.y)

        self.X_train, self.X_cal, self.y_train, self.y_cal = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=rs, stratify=y_temp)

        print(f"\nTrain: {len(self.X_train)} samples "
              f"({self.y_train.sum()} failures, {self.y_train.mean():.2%})")
        print(f"Cal  : {len(self.X_cal)} samples "
              f"({self.y_cal.sum()} failures, {self.y_cal.mean():.2%})")
        print(f"Test : {len(self.X_test)} samples "
              f"({self.y_test.sum()} failures, {self.y_test.mean():.2%})")

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def advanced_preprocessing(self, use_knn_imputer=False, use_robust_scaler=True,
                                pca_components=None, feature_selection_method="rfe",
                                remove_correlated=True, correlation_threshold=0.9,
                                add_lof_score=False):
        print("\nPreprocessing (leakage-safe) …")

        imp = KNNImputer(n_neighbors=5) if use_knn_imputer else SimpleImputer(strategy="median")
        self.X_train = pd.DataFrame(imp.fit_transform(self.X_train))
        self.X_cal   = pd.DataFrame(imp.transform(self.X_cal))
        self.X_test  = pd.DataFrame(imp.transform(self.X_test))

        const = [c for c in self.X_train.columns
                 if self.X_train[c].nunique() <= 1 or self.X_train[c].std() < 1e-6]
        if const:
            self.X_train.drop(columns=const, inplace=True)
            self.X_cal.drop(columns=const, inplace=True)
            self.X_test.drop(columns=const, inplace=True)
            print(f"  Removed {len(const)} constant columns")

        if remove_correlated:
            corr = self.X_train.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            hc = [c for c in upper.columns if any(upper[c] > correlation_threshold)]
            if hc:
                self.X_train.drop(columns=hc, inplace=True)
                self.X_cal.drop(columns=hc, inplace=True)
                self.X_test.drop(columns=hc, inplace=True)
                print(f"  Removed {len(hc)} correlated columns")

        scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train))
        self.X_cal   = pd.DataFrame(scaler.transform(self.X_cal))
        self.X_test  = pd.DataFrame(scaler.transform(self.X_test))

        n_sel = min(200, self.X_train.shape[1])
        if feature_selection_method == "rfe":
            print("  RFE feature selection …")
            sel = RFE(RandomForestClassifier(n_estimators=50, random_state=self.random_state),
                      n_features_to_select=n_sel)
            self.X_train = pd.DataFrame(sel.fit_transform(self.X_train, self.y_train))
            self.X_cal   = pd.DataFrame(sel.transform(self.X_cal))
            self.X_test  = pd.DataFrame(sel.transform(self.X_test))
        elif feature_selection_method == "kbest":
            sel = SelectKBest(f_classif, k=n_sel)
            self.X_train = pd.DataFrame(sel.fit_transform(self.X_train, self.y_train))
            self.X_cal   = pd.DataFrame(sel.transform(self.X_cal))
            self.X_test  = pd.DataFrame(sel.transform(self.X_test))
        elif feature_selection_method == "model_based":
            sel = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                threshold="median")
            self.X_train = pd.DataFrame(sel.fit_transform(self.X_train, self.y_train))
            self.X_cal   = pd.DataFrame(sel.transform(self.X_cal))
            self.X_test  = pd.DataFrame(sel.transform(self.X_test))

        if pca_components:
            pca = PCA(n_components=pca_components)
            self.X_train = pd.DataFrame(pca.fit_transform(self.X_train))
            self.X_cal   = pd.DataFrame(pca.transform(self.X_cal))
            self.X_test  = pd.DataFrame(pca.transform(self.X_test))
            print(f"  PCA variance: {pca.explained_variance_ratio_.sum():.3f}")

        if add_lof_score:
            lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
            lof.fit(self.X_train)
            self.X_train["lof_score"] = -lof.score_samples(self.X_train)
            self.X_cal["lof_score"]   = -lof.score_samples(self.X_cal)
            self.X_test["lof_score"]  = -lof.score_samples(self.X_test)
            self.X_train.columns = range(self.X_train.shape[1])
            self.X_cal.columns   = range(self.X_cal.shape[1])
            self.X_test.columns  = range(self.X_test.shape[1])

        self.feature_names = [f"Feature_{i}" for i in range(self.X_train.shape[1])]
        print(f"  Final shape — train: {self.X_train.shape}  "
              f"cal: {self.X_cal.shape}  test: {self.X_test.shape}")
        print(f"  Missing — train: {self.X_train.isnull().sum().sum()}  "
              f"cal: {self.X_cal.isnull().sum().sum()}  "
              f"test: {self.X_test.isnull().sum().sum()}")

    # ── Model Registry ────────────────────────────────────────────────────────

    def get_advanced_models(self) -> dict:
        models = {
            "Baseline LR": LogisticRegression(
                random_state=self.random_state, max_iter=1000,
                class_weight="balanced", penalty="l2", solver="lbfgs"),
            "Baseline Decision Tree": DecisionTreeClassifier(
                random_state=self.random_state, class_weight="balanced", max_depth=10),
            "Balanced Random Forest": BalancedRandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=self.random_state),
            "Balanced Bagging": BalancedBaggingClassifier(
                n_estimators=100, random_state=self.random_state),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=200, max_depth=15, random_state=self.random_state,
                class_weight="balanced"),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=8,
                random_state=self.random_state, subsample=0.8, min_samples_leaf=5),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=100, learning_rate=0.8, random_state=self.random_state),
            "SVM RBF": SVC(
                kernel="rbf", probability=True, random_state=self.random_state,
                class_weight="balanced"),
            "Neural Network": MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500,
                random_state=self.random_state,
                early_stopping=True, validation_fraction=0.1),
            "Logistic Regression": LogisticRegression(
                random_state=self.random_state, max_iter=1000,
                class_weight="balanced", penalty="elasticnet",
                l1_ratio=0.5, solver="saga"),
            "K-Neighbors": KNeighborsClassifier(n_neighbors=7, weights="distance"),
            "Gaussian NB": GaussianNB(),
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                scale_pos_weight=float(self.class_weight_ratio), eval_metric="logloss",
                random_state=self.random_state, subsample=0.8, colsample_bytree=0.8)
        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=6, num_leaves=31,
                scale_pos_weight=float(self.class_weight_ratio),
                random_state=self.random_state, subsample=0.8, colsample_bytree=0.8,
                verbosity=-1)
        if CATBOOST_AVAILABLE:
            models["CatBoost"] = cb.CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=6,
                scale_pos_weight=float(self.class_weight_ratio),
                random_seed=self.random_state, verbose=False,
                eval_metric="F1", loss_function="Logloss")
        return models

    def get_advanced_samplers(self) -> dict:
        return {
            "Original (cost-sensitive)": None,
            "SMOTE":                SMOTE(random_state=self.random_state, k_neighbors=3),
            "Borderline SMOTE":     BorderlineSMOTE(random_state=self.random_state,
                                                    kind="borderline-1"),
            "SVM SMOTE":            SVMSMOTE(random_state=self.random_state),
            "ADASYN":               ADASYN(random_state=self.random_state),
            "SMOTE + Tomek":        SMOTETomek(random_state=self.random_state),
            "SMOTE + ENN":          SMOTEENN(random_state=self.random_state),
            "Random Undersampling": RandomUnderSampler(random_state=self.random_state,
                                                       sampling_strategy=0.3),
        }

    # ── Calibration & Threshold ───────────────────────────────────────────────

    def optimize_threshold(self, y_true, y_proba) -> tuple[float, float]:
        """Search [0.05, 0.95] for best F1 on calibrated probabilities."""
        best_score, best_thr = 0.0, 0.5
        for thr in np.arange(0.05, 0.95, 0.01):
            pred = (y_proba >= thr).astype(int)
            score = f1_score(y_true, pred, zero_division=0)
            if score > best_score:
                best_score, best_thr = score, thr
        return best_thr, best_score

    def _calibrate_and_threshold(self, model, X_cal, y_cal):
        """Leakage-free probability calibration using held-out calibration set."""
        if not hasattr(model, "predict_proba"):
            return model, 0.5

        try:
            X_cal_fit, X_cal_val, y_cal_fit, y_cal_val = train_test_split(
                X_cal, y_cal,
                test_size=0.50, random_state=self.random_state, stratify=y_cal)
        except ValueError:
            try:
                raw_proba = model.predict_proba(X_cal)[:, 1]
                best_thr, _ = self.optimize_threshold(y_cal, raw_proba)
            except Exception:
                best_thr = 0.5
            return model, best_thr

        try:
            try:
                from sklearn.frozen import FrozenEstimator            # sklearn >= 1.6
                cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
            except ImportError:
                cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
            cal.fit(X_cal_fit, y_cal_fit)
            val_proba = cal.predict_proba(X_cal_val)[:, 1]
            best_thr, _ = self.optimize_threshold(y_cal_val, val_proba)
            return cal, best_thr
        except Exception as e:
            print(f"    [WARN] Calibration failed ({e}); using UNCALIBRATED model. "
                  "Reported probabilities are NOT calibrated.")
            try:
                raw_proba = model.predict_proba(X_cal_val)[:, 1]
                best_thr, _ = self.optimize_threshold(y_cal_val, raw_proba)
            except Exception:
                best_thr = 0.5
            return model, best_thr

    # ── Training ──────────────────────────────────────────────────────────────

    def _evaluate_and_store(self, key, model, model_name, sampling_method,
                             X_resampled, y_resampled, sampler=None):
        model = clone(model)
        model.fit(X_resampled, y_resampled)

        predict_model, best_thr = self._calibrate_and_threshold(model, self.X_cal, self.y_cal)

        _lats = []
        _s = self.X_test.iloc[:1]
        _predict_fn = (predict_model.predict_proba
                       if hasattr(predict_model, "predict_proba")
                       else predict_model.predict)
        for _ in range(10):
            _t = time.perf_counter()
            _predict_fn(_s)
            _lats.append((time.perf_counter() - _t) * 1000)
        latency_ms = float(np.median(_lats))

        y_proba = (predict_model.predict_proba(self.X_test)[:, 1]
                   if hasattr(predict_model, "predict_proba") else None)

        y_pred = ((y_proba >= best_thr).astype(int)
                  if y_proba is not None else predict_model.predict(self.X_test))

        is_degen = int(y_pred.sum()) < 2
        if is_degen:
            print(f"    [DEGEN] Only {int(y_pred.sum())} positive(s) — flagged, "
                  "excluded from rankings.")

        acc   = accuracy_score(self.y_test, y_pred)
        prec  = precision_score(self.y_test, y_pred, zero_division=0)
        rec   = recall_score(self.y_test, y_pred, zero_division=0)
        f1    = f1_score(self.y_test, y_pred, zero_division=0)
        mcc   = matthews_corrcoef(self.y_test, y_pred)
        roc   = roc_auc_score(self.y_test, y_proba) if y_proba is not None else None
        prauc = average_precision_score(self.y_test, y_proba) if y_proba is not None else None

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        if sampler is not None:
            cv_pipe = ImbPipeline([("sampler", clone(sampler)),
                                   ("model",   clone(model))])
            cv_scores = cross_val_score(cv_pipe, self.X_train, self.y_train,
                                        cv=skf, scoring="f1")
        else:
            cv_scores = cross_val_score(clone(model), self.X_train, self.y_train,
                                        cv=skf, scoring="f1")

        ci = self.bootstrap_metric_cis(
            self.y_test.values, y_pred, y_proba,
            n_boot=1000, random_state=self.random_state)

        self.results[key] = dict(
            model=model, predict_model=predict_model,
            sampling_method=sampling_method, model_name=model_name,
            predictions=y_pred, predictions_proba=y_proba,
            best_threshold=best_thr, is_degenerate=is_degen,
            accuracy=acc, precision=prec, recall=rec,
            f1_score=f1, mcc=mcc, roc_auc=roc, pr_auc=prauc,
            cv_scores=cv_scores, cv_mean=cv_scores.mean(), cv_std=cv_scores.std(),
            inference_latency_ms=latency_ms, ci=ci,
        )

        f1_ci = ci.get("f1")
        ci_str = (f"[{f1_ci[0]:.3f}–{f1_ci[1]:.3f}]" if f1_ci else "")
        tag    = " [DEGEN]" if is_degen else ""
        print(f"  F1: {f1:.4f} {ci_str} | MCC: {mcc:.4f} | "
              f"PR-AUC: {self._fmt(prauc)} | CV-F1: {cv_scores.mean():.4f} | "
              f"Thr: {best_thr:.2f} | Lat: {latency_ms:.1f} ms{tag}")

    def create_ensemble_models(self, base_models, X_train, y_train) -> dict:
        ml  = list(base_models.items())[:5]
        ens = {"Voting Ensemble": VotingClassifier(estimators=ml, voting="soft")}
        try:
            ens["Stacking Ensemble"] = StackingClassifier(
                estimators=ml,
                final_estimator=LogisticRegression(class_weight="balanced"),
                cv=3)
        except Exception as e:
            print(f"StackingClassifier skipped: {e}")
        return ens

    def train_advanced_models(self):
        print("\nTraining advanced models …")
        base_models = self.get_advanced_models()
        samplers    = self.get_advanced_samplers()

        for sname, sampler in samplers.items():
            print(f"\n{'─'*60}\nSampler: {sname}\n{'─'*60}")
            if sampler is None:
                X_res, y_res = self.X_train.copy(), self.y_train.copy()
            else:
                try:
                    X_res, y_res = sampler.fit_resample(self.X_train, self.y_train)
                    print(f"  Resampled: {X_res.shape} | "
                          f"Dist: {pd.Series(y_res).value_counts().to_dict()}")
                except Exception as e:
                    print(f"  Sampling failed: {e}"); continue

            for mname, model in base_models.items():
                print(f"\nTraining {mname} …")
                try:
                    self._evaluate_and_store(
                        key=f"{sname}_{mname}", model=model, model_name=mname,
                        sampling_method=sname,
                        X_resampled=X_res, y_resampled=y_res, sampler=sampler)
                except Exception as e:
                    print(f"  ✗ {e}"); traceback.print_exc()

        best_sname = self.find_best_sampler()
        sampler    = samplers[best_sname]
        if sampler is None:
            X_res, y_res = self.X_train.copy(), self.y_train.copy()
        else:
            X_res, y_res = sampler.fit_resample(self.X_train, self.y_train)

        print(f"\n{'─'*60}\nEnsemble models (sampler: {best_sname})\n{'─'*60}")
        for ename, emodel in self.create_ensemble_models(base_models, X_res, y_res).items():
            print(f"\nTraining {ename} …")
            try:
                self._evaluate_and_store(
                    key=f"{best_sname}_{ename}", model=emodel, model_name=ename,
                    sampling_method=best_sname,
                    X_resampled=X_res, y_resampled=y_res, sampler=sampler)
            except Exception as e:
                print(f"  ✗ {e}"); traceback.print_exc()

    def find_best_sampler(self) -> str:
        if not self.results:
            return "SMOTE"
        scores: dict[str, list] = {}
        for r in self.results.values():
            if not r.get("is_degenerate", False):
                scores.setdefault(r["sampling_method"], []).append(r["f1_score"])
        if not scores:
            return "SMOTE"
        return max(scores, key=lambda s: np.mean(scores[s]))

    # ── Hyperparameter Tuning ─────────────────────────────────────────────────

    def hyperparameter_optimization(self, top_models: int = 3):
        print(f"\nOptimising hyperparameters for top {top_models} models …")

        nd_df = pd.DataFrame([
            {"key": k, "sampling_method": v["sampling_method"],
             "model_name": v["model_name"], "f1_score": v["f1_score"]}
            for k, v in self.results.items()
            if not v.get("is_degenerate", False)
        ]).sort_values("f1_score", ascending=False)

        param_grids = {
            "Balanced Random Forest": {
                "n_estimators": [100, 200], "max_depth": [10, 15], "min_samples_split": [2, 5]},
            "Gradient Boosting": {
                "n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [6, 8]},
            "Extra Trees": {
                "n_estimators": [100, 200], "max_depth": [10, 15]},
            "XGBoost": {
                "n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                "max_depth": [4, 6]},
            "LightGBM": {
                "n_estimators": [200, 300], "learning_rate": [0.01, 0.05],
                "num_leaves": [31, 63]},
            "CatBoost": {
                "iterations": [200, 300], "learning_rate": [0.01, 0.05], "depth": [4, 6]},
        }

        samplers = self.get_advanced_samplers()
        for key in nd_df.head(top_models)["key"].tolist():
            res = self.results[key]
            mn, sm = res["model_name"], res["sampling_method"]
            if mn not in param_grids:
                continue

            print(f"\nOptimising {mn} with {sm} …")
            sampler = samplers.get(sm)

            skf = StratifiedKFold(n_splits=3, shuffle=True,
                                  random_state=self.random_state)
            grid = {f"model__{k}": v for k, v in param_grids[mn].items()}
            if sampler is not None:
                est = ImbPipeline([("sampler", clone(sampler)),
                                   ("model",   self.get_advanced_models()[mn])])
            else:
                est = ImbPipeline([("model", self.get_advanced_models()[mn])])
            search = GridSearchCV(est, grid, cv=skf,
                                  scoring="average_precision", n_jobs=-1, verbose=0)
            search.fit(self.X_train, self.y_train)

            best_params = {k.replace("model__", ""): v
                           for k, v in search.best_params_.items()}
            tuned_model = self.get_advanced_models()[mn].set_params(**best_params)
            X_res, y_res = self.X_train.copy(), self.y_train.copy()
            if sampler is not None:
                X_res, y_res = clone(sampler).fit_resample(X_res, y_res)

            opt_key = f"{key}_OPTIMIZED"
            try:
                self._evaluate_and_store(
                    key=opt_key, model=tuned_model,
                    model_name=f"{mn} (Optimized)", sampling_method=sm,
                    X_resampled=X_res, y_resampled=y_res, sampler=sampler)
                self.results[opt_key]["best_params"] = best_params
                print(f"  Best params : {search.best_params_}")
                print(f"  Opt F1      : {self.results[opt_key]['f1_score']:.4f} "
                      f"(orig: {res['f1_score']:.4f})")
            except Exception as e:
                print(f"  ✗ {e}"); traceback.print_exc()

    # ── Bootstrap CIs & McNemar ───────────────────────────────────────────────

    def bootstrap_metric_cis(self, y_true, y_pred, y_proba=None,
                              n_boot: int = 1000, alpha: float = 0.05,
                              random_state=None) -> dict:
        rng = np.random.default_rng(random_state)
        n   = len(y_true)
        f1s, mccs, aucs, praucs = [], [], [], []

        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            yt, yp = y_true[idx], y_pred[idx]
            if len(np.unique(yt)) < 2:
                continue
            f1s.append(f1_score(yt, yp, zero_division=0))
            mccs.append(matthews_corrcoef(yt, yp))
            if y_proba is not None:
                ypr = y_proba[idx]
                try:
                    aucs.append(roc_auc_score(yt, ypr))
                    praucs.append(average_precision_score(yt, ypr))
                except Exception:
                    pass

        def _ci(vals):
            if len(vals) < 10:
                return None
            return (float(np.percentile(vals, 100 * alpha / 2)),
                    float(np.percentile(vals, 100 * (1 - alpha / 2))))

        return {"f1": _ci(f1s), "mcc": _ci(mccs),
                "roc_auc": _ci(aucs)   if aucs   else None,
                "pr_auc":  _ci(praucs) if praucs else None,
                "n_boot_effective": len(f1s)}

    def mcnemar_compare(self, key_a: str, key_b: str, correction: bool = True) -> dict:
        if key_a not in self.results or key_b not in self.results:
            raise KeyError("Both keys must be in self.results.")

        y_true  = np.asarray(self.y_test)
        pred_a  = self.results[key_a]["predictions"]
        pred_b  = self.results[key_b]["predictions"]
        corr_a  = pred_a == y_true
        corr_b  = pred_b == y_true
        n10 = int(np.sum(corr_a & ~corr_b))
        n01 = int(np.sum(~corr_a & corr_b))
        n11 = int(np.sum(corr_a & corr_b))
        n00 = int(np.sum(~corr_a & ~corr_b))
        table = np.array([[n11, n10], [n01, n00]])

        result = dict(
            model_a=self.results[key_a]["model_name"],
            model_b=self.results[key_b]["model_name"],
            contingency_table=table, n_a_only=n10, n_b_only=n01)

        if STATSMODELS_AVAILABLE:
            mc = _sm_mcnemar(table, exact=(n10 + n01) < 25, correction=correction)
            result["statistic"] = mc.statistic
            result["p_value"]   = mc.pvalue
        elif SCIPY_AVAILABLE:
            b, c = n10, n01
            if b + c == 0:
                stat, p = 0.0, 1.0
            else:
                num  = (abs(b - c) - 1) ** 2 if correction else (b - c) ** 2
                stat = num / (b + c)
                p    = 1 - _chi2.cdf(stat, df=1)
            result["statistic"] = stat
            result["p_value"]   = p
        else:
            result["statistic"] = None
            result["p_value"]   = None
            print("  SciPy/statsmodels not available — p-value cannot be computed.")

        sig = (result["p_value"] or 1.0) < 0.05
        print(f"\nMcNemar: {result['model_a']} vs {result['model_b']}")
        print(f"  A-only={n10} | B-only={n01} | "
              f"stat={self._fmt(result['statistic'])} | "
              f"p={self._fmt(result['p_value'])} | "
              f"{'✓ Significant' if sig else '✗ n.s.'}")
        return result

    def mcnemar_top_models(self, top_n: int = 5) -> pd.DataFrame:
        nd = [(k, v) for k, v in self.results.items()
              if not v.get("is_degenerate", False)]
        top_keys = [k for k, _ in sorted(nd, key=lambda x: x[1]["f1_score"],
                                          reverse=True)[:top_n]]
        rows = []
        for i in range(len(top_keys)):
            for j in range(i + 1, len(top_keys)):
                r = self.mcnemar_compare(top_keys[i], top_keys[j])
                rows.append({
                    "Model A": r["model_a"], "Model B": r["model_b"],
                    "A-only": r["n_a_only"], "B-only": r["n_b_only"],
                    "Stat": r.get("statistic"), "p-value": r.get("p_value"),
                    "Sig (p<0.05)": (r.get("p_value") or 1.0) < 0.05,
                })
        return pd.DataFrame(rows)

    # ── Reports ───────────────────────────────────────────────────────────────

    def _summary_df(self) -> pd.DataFrame:
        rows = []
        for key, r in self.results.items():
            ci   = r.get("ci", {})
            f1ci = ci.get("f1")
            rows.append(dict(
                Key=key, Sampling=r["sampling_method"], Model=r["model_name"],
                Accuracy=r["accuracy"], Precision=r["precision"], Recall=r["recall"],
                F1=r["f1_score"],
                F1_lo=f1ci[0] if f1ci else None,
                F1_hi=f1ci[1] if f1ci else None,
                MCC=r["mcc"], ROC_AUC=r["roc_auc"], PR_AUC=r["pr_auc"],
                CV_F1=r.get("cv_mean"), CV_std=r.get("cv_std"),
                Threshold=r.get("best_threshold", 0.5),
                Latency_ms=r.get("inference_latency_ms"),
                Degenerate=r.get("is_degenerate", False),
            ))
        return pd.DataFrame(rows).sort_values("F1", ascending=False)

    def show_comprehensive_results(self) -> pd.DataFrame:
        print("\n" + "=" * 110)
        print("COMPREHENSIVE ADVANCED RESULTS")
        print("=" * 110)

        df = self._summary_df()
        nd = df[~df["Degenerate"]]

        cols = ["Key", "Sampling", "Model", "F1", "F1_lo", "F1_hi",
                "MCC", "PR_AUC", "CV_F1", "CV_std", "Threshold", "Latency_ms"]
        print("\nTop 15 non-degenerate configurations:")
        print(nd[cols].head(15).to_string(index=False))

        n_degen = df["Degenerate"].sum()
        if n_degen:
            print(f"\n[Note] {n_degen} degenerate model(s) excluded.")

        if df.empty:
            print("No models to report."); return df
        best = nd.iloc[0] if len(nd) else df.iloc[0]
        print("\n🏆 BEST CONFIGURATION:")
        for c in ["Model", "Sampling", "F1", "F1_lo", "F1_hi",
                  "MCC", "Precision", "Recall", "ROC_AUC", "PR_AUC",
                  "CV_F1", "Threshold", "Latency_ms"]:
            print(f"  {c}: {best[c]}")

        print("\n📊 PERFORMANCE INSIGHTS:")
        print(f"  Max F1 (non-degen): {nd['F1'].max():.4f}")
        print(f"  Avg F1 (non-degen): {nd['F1'].mean():.4f}")
        print(f"  Avg MCC            : {nd['MCC'].mean():.4f}")

        samp = nd.groupby("Sampling")["F1"].agg(["mean", "max", "count"])
        print(f"\n🎯 BEST SAMPLING METHODS:\n"
              f"{samp.sort_values('mean', ascending=False).head()}")

        self.create_advanced_visualizations(df)
        return df

    def show_detailed_classification_reports(self):
        print("\n" + "=" * 80)
        print("DETAILED CLASSIFICATION REPORTS — TOP 10 NON-DEGENERATE MODELS")
        print("=" * 80)
        nd = sorted(
            [(k, v) for k, v in self.results.items()
             if not v.get("is_degenerate", False)],
            key=lambda x: x[1]["f1_score"], reverse=True)
        for i, (key, res) in enumerate(nd[:10]):
            ci  = res.get("ci", {}).get("f1")
            ci_str = (f" [{ci[0]:.3f}–{ci[1]:.3f}]" if ci else "")
            print(f"\n{i+1}. {res['model_name']} ({res['sampling_method']})")
            print(f"   F1: {res['f1_score']:.4f}{ci_str} | MCC: {res['mcc']:.4f} | "
                  f"CV-F1: {res.get('cv_mean', 0):.4f} | "
                  f"Thr: {res.get('best_threshold', 0.5):.3f}")
            print("-" * 70)
            print(classification_report(self.y_test, res["predictions"],
                                        target_names=["PASS", "FAIL"], digits=4))
            cm = confusion_matrix(self.y_test, res["predictions"])
            print("Confusion Matrix:")
            print("              Pred PASS  Pred FAIL")
            print(f"Actual PASS    {cm[0,0]:5d}      {cm[0,1]:5d}")
            print(f"Actual FAIL    {cm[1,0]:5d}      {cm[1,1]:5d}\n")

    def analyze_class_imbalance_impact(self):
        print("\n" + "=" * 80)
        print("CLASS IMBALANCE IMPACT ANALYSIS")
        print("=" * 80)
        analysis: dict = {}
        for r in self.results.values():
            if r.get("is_degenerate", False):
                continue
            sm = r["sampling_method"]
            cr = classification_report(self.y_test, r["predictions"],
                                       target_names=["PASS", "FAIL"], output_dict=True)
            d  = analysis.setdefault(sm, dict(f1=[], mcc=[], pf=[], rf=[], pp=[], rp=[]))
            d["f1"].append(r["f1_score"]); d["mcc"].append(r["mcc"])
            d["pf"].append(cr["FAIL"]["precision"]); d["rf"].append(cr["FAIL"]["recall"])
            d["pp"].append(cr["PASS"]["precision"]); d["rp"].append(cr["PASS"]["recall"])
        print("\nAverage by sampling method (non-degenerate):")
        for sm, d in analysis.items():
            print(f"\n  {sm}:")
            print(f"    F1  : {np.mean(d['f1']):.4f} | MCC: {np.mean(d['mcc']):.4f}")
            print(f"    FAIL: P={np.mean(d['pf']):.4f}  R={np.mean(d['rf']):.4f}")
            print(f"    PASS: P={np.mean(d['pp']):.4f}  R={np.mean(d['rp']):.4f}")

    def generate_final_report(self):
        print("\n" + "=" * 80)
        print("FINAL COMPREHENSIVE REPORT")
        print("=" * 80)

        df   = self.show_comprehensive_results()
        nd   = df[~df["Degenerate"]]
        best = nd.iloc[0] if len(nd) else df.iloc[0]
        bkey = best["Key"]
        bres = self.results[bkey]

        f1ci = (f"[{best['F1_lo']:.3f}–{best['F1_hi']:.3f}]"
                if best["F1_lo"] is not None else "N/A")
        print(f"\n🎯 PRODUCTION RECOMMENDATION:")
        print(f"  Model     : {best['Model']}")
        print(f"  Sampler   : {best['Sampling']}")
        print(f"  F1        : {best['F1']:.4f}  95% CI {f1ci}")
        print(f"  MCC       : {best['MCC']:.4f}")
        print(f"  PR-AUC    : {self._fmt(best['PR_AUC'])}")
        print(f"  CV-F1     : {self._fmt(best['CV_F1'], '.4f')} ± "
              f"{self._fmt(best['CV_std'], '.4f')}  (leakage-free)")
        print(f"  Threshold : {best['Threshold']:.3f}  "
              f"(calibrated on real class distribution)")
        print(f"  Latency   : {self._fmt(best['Latency_ms'], '.1f')} ms")

        tn, fp, fn, tp = confusion_matrix(self.y_test, bres["predictions"]).ravel()
        total    = fp * self.cost_fp + fn * self.cost_fn
        baseline = int(self.y_test.sum()) * self.cost_fn
        print(f"\n💼 BUSINESS IMPACT:")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  Cost (model)   : ${total:,}")
        print(f"  Baseline cost  : ${baseline:,}")
        print(f"  Savings        : ${baseline - total:,}")

        print(f"\n📐 STATISTICAL SIGNIFICANCE (McNemar top 5):")
        try:
            print(self.mcnemar_top_models(top_n=5).to_string(index=False))
        except Exception as e:
            print(f"  Skipped: {e}")

        print(f"\n🔧 IMPLEMENTATION GUIDELINES:")
        print(f"  1. Train with '{best['Sampling']}' for data preparation.")
        print(f"  2. Wrap sampler + model in ImbPipeline for deployment.")
        print(f"  3. Calibrate probabilities with CalibratedClassifierCV "
              f"(isotonic, cv='prefit') on original distribution.")
        print(f"  4. Set threshold to {best['Threshold']:.3f} on calibrated probs.")
        print(f"  5. Report F1 with 95% bootstrap CI, not as a bare number.")
        print(f"  6. Run KS-test drift detection (Bonferroni) on each batch.")
        print(f"  7. Use SHAP TreeExplainer for per-prediction explanations.")

        model_path = self.output_dir / "secom_best_model.pkl"
        joblib.dump(bres["model"], model_path)
        print(f"\n💾 Model saved → {model_path}")
        return bkey, bres

    # ── Plots ─────────────────────────────────────────────────────────────────

    def create_advanced_visualizations(self, df: pd.DataFrame):
        nd = df[~df["Degenerate"]]
        if nd.empty:
            print("No non-degenerate models to plot."); return

        fig, axes = plt.subplots(2, 3, figsize=(22, 12))

        top10 = nd.head(10)
        f1_lo = pd.to_numeric(top10["F1_lo"], errors="coerce").fillna(top10["F1"]).values
        f1_hi = pd.to_numeric(top10["F1_hi"], errors="coerce").fillna(top10["F1"]).values
        ci_lo = np.abs(top10["F1"].values - f1_lo)
        ci_hi = np.abs(f1_hi - top10["F1"].values)
        axes[0, 0].barh(range(len(top10)), top10["F1"])
        axes[0, 0].errorbar(top10["F1"], range(len(top10)),
                            xerr=[ci_lo, ci_hi], fmt="none", color="black", capsize=3)
        axes[0, 0].set_yticks(range(len(top10)))
        axes[0, 0].set_yticklabels(
            [f"{r['Model'][:18]} ({r['Sampling'][:8]})" for _, r in top10.iterrows()],
            fontsize=7)
        axes[0, 0].set_xlabel("F1 (95% CI)"); axes[0, 0].set_title("Top 10 by F1")
        axes[0, 0].grid(True, alpha=0.3)

        samp = nd.groupby("Sampling")["F1"].mean().sort_values(ascending=False)
        axes[0, 1].bar(range(len(samp)), samp.values)
        axes[0, 1].set_xticks(range(len(samp)))
        axes[0, 1].set_xticklabels(samp.index, rotation=45, ha="right", fontsize=7)
        axes[0, 1].set_ylabel("Avg F1"); axes[0, 1].set_title("Sampling Methods")
        axes[0, 1].grid(True, alpha=0.3)

        sc = axes[0, 2].scatter(nd["Precision"], nd["Recall"],
                                c=nd["F1"], cmap="viridis", alpha=0.7)
        plt.colorbar(sc, ax=axes[0, 2], label="F1")
        axes[0, 2].set_xlabel("Precision"); axes[0, 2].set_ylabel("Recall")
        axes[0, 2].set_title("Precision vs Recall"); axes[0, 2].grid(True, alpha=0.3)

        mp = nd.groupby("Model")["F1"].mean().sort_values(ascending=False)
        axes[1, 0].bar(range(len(mp)), mp.values)
        axes[1, 0].set_xticks(range(len(mp)))
        axes[1, 0].set_xticklabels(mp.index, rotation=45, ha="right", fontsize=7)
        axes[1, 0].set_ylabel("Avg F1"); axes[1, 0].set_title("Model Performance")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].scatter(nd["CV_F1"], nd["F1"], alpha=0.6)
        lim = max(nd["F1"].max(), nd["CV_F1"].fillna(0).max()) + 0.05
        axes[1, 1].plot([0, lim], [0, lim], "k--", alpha=0.4, label="y=x")
        axes[1, 1].set_xlabel("CV F1 (leakage-free)"); axes[1, 1].set_ylabel("Test F1")
        axes[1, 1].set_title("CV vs Test F1 (gap)"); axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        lat = nd.dropna(subset=["Latency_ms"]).head(10)
        axes[1, 2].barh(range(len(lat)), lat["Latency_ms"])
        axes[1, 2].set_yticks(range(len(lat)))
        axes[1, 2].set_yticklabels(lat["Model"], fontsize=7)
        axes[1, 2].set_xlabel("Latency (ms)"); axes[1, 2].set_title("Inference Latency")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "secom_performance_overview.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

        best_key = nd.iloc[0]["Key"]
        br       = self.results[best_key]
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, br["predictions"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Failure"],
                    yticklabels=["Normal", "Failure"])
        plt.title(f"Confusion Matrix — {br['model_name']} ({br['sampling_method']})")
        plt.ylabel("True"); plt.xlabel("Predicted")
        plt.savefig(self.output_dir / "secom_confusion_matrix.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

        plt.figure(figsize=(10, 8))
        for _, row in nd.head(5).iterrows():
            r = self.results[row["Key"]]
            if r["predictions_proba"] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, r["predictions_proba"])
                plt.plot(fpr, tpr,
                         label=f"{r['model_name']} AUC={self._fmt(r['roc_auc'], '.3f')}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves — Top 5")
        plt.legend(fontsize=8); plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "secom_roc_curves.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

        plt.figure(figsize=(10, 8))
        for _, row in nd.head(5).iterrows():
            r = self.results[row["Key"]]
            if r["predictions_proba"] is not None:
                prec_arr, rec_arr, _ = precision_recall_curve(self.y_test, r["predictions_proba"])
                plt.plot(rec_arr, prec_arr,
                         label=f"{r['model_name']} PR-AUC={self._fmt(r['pr_auc'], '.3f')}")
        base = float(self.y_test.mean())
        plt.axhline(y=base, color="k", linestyle="--", label=f"Baseline ({base:.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision-Recall Curves — Top 5")
        plt.legend(fontsize=8); plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "secom_pr_curves.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

    def plot_learning_curves(self):
        print("\nPlotting learning curves …")
        nd = [(k, v) for k, v in self.results.items()
              if not v.get("is_degenerate", False)]
        if not nd:
            print("No non-degenerate models."); return
        best_key, best_res = sorted(nd, key=lambda x: x[1]["f1_score"], reverse=True)[0]
        model = best_res["model"]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        sizes, tr_sc, val_sc = learning_curve(
            model, self.X_train, self.y_train,
            train_sizes=np.linspace(0.1, 1.0, 8),
            cv=skf, scoring="f1", n_jobs=-1)

        plt.figure(figsize=(10, 6))
        plt.plot(sizes, tr_sc.mean(1), "o-", label="Train F1")
        plt.fill_between(sizes, tr_sc.mean(1) - tr_sc.std(1),
                         tr_sc.mean(1) + tr_sc.std(1), alpha=0.15)
        plt.plot(sizes, val_sc.mean(1), "s-", label="Val F1")
        plt.fill_between(sizes, val_sc.mean(1) - val_sc.std(1),
                         val_sc.mean(1) + val_sc.std(1), alpha=0.15)
        plt.xlabel("Train size"); plt.ylabel("F1"); plt.title(f"Learning Curve — {best_key}")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "secom_learning_curve.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

    def feature_importance_analysis(self):
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        nd = [(k, v) for k, v in self.results.items()
              if not v.get("is_degenerate", False)]
        if not nd:
            print("No non-degenerate models."); return
        best_key, best_res = sorted(nd, key=lambda x: x[1]["f1_score"], reverse=True)[0]
        model = best_res["model"]
        print(f"Analysing: {best_res['model_name']}")

        if hasattr(model, "feature_importances_"):
            imp   = model.feature_importances_
            top_n = min(20, len(imp))
            idx   = np.argsort(imp)[::-1][:top_n]
            for rank, i in enumerate(idx):
                print(f"  {rank+1:2d}. Feature_{i}: {imp[i]:.4f}")
            plt.figure(figsize=(12, 8))
            plt.barh(range(top_n), imp[idx])
            plt.yticks(range(top_n), [f"Feature_{i}" for i in idx])
            plt.xlabel("Importance"); plt.gca().invert_yaxis()
            plt.title(f"Feature Importances — {best_res['model_name']}")
            plt.tight_layout()
            plt.savefig(self.output_dir / "secom_feature_importance.png",
                        dpi=150, bbox_inches="tight")
            plt.show()
        else:
            print("Model does not expose feature_importances_. Run SHAP instead.")

    # ── Explainability (SHAP / LIME) ──────────────────────────────────────────

    def shap_analysis(self, max_display: int = 20):
        if not SHAP_AVAILABLE:
            print("SHAP not installed — pip install shap"); return

        nd = [(k, v) for k, v in self.results.items()
              if not v.get("is_degenerate", False)]
        if not nd:
            print("No non-degenerate models for SHAP."); return
        best_key, best_res = sorted(nd, key=lambda x: x[1]["f1_score"], reverse=True)[0]

        model      = best_res["model"]
        model_name = best_res["model_name"]
        print(f"\nRunning SHAP on: {model_name}")

        X_arr = (self.X_test.values
                 if hasattr(self.X_test, "values") else np.asarray(self.X_test))
        X_shap = X_arr

        try:
            try:
                explainer   = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_arr)
                print("  Using TreeExplainer.")
            except Exception as te:
                print(f"  TreeExplainer unsupported ({te}); "
                      "using KernelExplainer (slower) …")
                bg          = shap.sample(self.X_train, min(100, len(self.X_train)))
                explainer   = shap.KernelExplainer(model.predict_proba, bg)
                X_shap      = X_arr[:50]
                shap_values = explainer.shap_values(X_shap)

            if isinstance(shap_values, list):
                sv = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                sv = shap_values[:, :, 1]
            else:
                sv = shap_values

            plt.figure(figsize=(12, 8))
            shap.summary_plot(sv, X_shap, feature_names=self.feature_names,
                              max_display=max_display, show=False)
            plt.title(f"SHAP Summary — {model_name}")
            plt.tight_layout()
            plt.savefig(self.output_dir / "secom_shap_summary.png",
                        dpi=150, bbox_inches="tight")
            plt.show()

            plt.figure(figsize=(10, 6))
            shap.summary_plot(sv, X_shap, feature_names=self.feature_names,
                              plot_type="bar", max_display=max_display, show=False)
            plt.title(f"SHAP Mean |Value| — {model_name}")
            plt.tight_layout()
            plt.savefig(self.output_dir / "secom_shap_bar.png",
                        dpi=150, bbox_inches="tight")
            plt.show()

            try:
                ev = explainer.expected_value
                base_val = (ev[1] if isinstance(ev, (list, np.ndarray)) else float(ev))
                exp = shap.Explanation(
                    values=sv[0], base_values=base_val,
                    data=X_arr[0], feature_names=self.feature_names)
                plt.figure()
                shap.plots.waterfall(exp, max_display=15, show=False)
                plt.title(f"SHAP Waterfall — sample 0 ({model_name})")
                plt.tight_layout()
                plt.savefig(self.output_dir / "secom_shap_waterfall.png",
                            dpi=150, bbox_inches="tight")
                plt.show()
            except Exception as we:
                print(f"  Waterfall skipped: {we}")

            print("SHAP plots saved.")
        except Exception as e:
            print(f"SHAP failed: {e}"); traceback.print_exc()

    def lime_analysis(self, sample_idx: int = 0):
        if not LIME_AVAILABLE:
            print("LIME not installed — pip install lime"); return
        nd = [(k, v) for k, v in self.results.items()
              if not v.get("is_degenerate", False)]
        if not nd:
            print("No non-degenerate models for LIME."); return
        model = sorted(nd, key=lambda x: x[1]["f1_score"], reverse=True)[0][1]["model"]
        print(f"\nRunning LIME (sample {sample_idx}) …")
        try:
            exp = LimeTabularExplainer(
                self.X_train.values, feature_names=self.feature_names,
                class_names=["PASS", "FAIL"], discretize_continuous=True,
            ).explain_instance(self.X_test.iloc[sample_idx].values,
                               model.predict_proba, num_features=15)
            exp.as_pyplot_figure()
            plt.title(f"LIME — sample {sample_idx}")
            plt.tight_layout()
            plt.savefig(self.output_dir / "secom_lime.png",
                        dpi=150, bbox_inches="tight")
            plt.show()
            print("LIME plot saved.")
        except Exception as e:
            print(f"LIME failed: {e}"); traceback.print_exc()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def anomaly_detection_analysis(self):
        print("\n" + "=" * 60)
        print("ANOMALY DETECTION ANALYSIS")
        print("=" * 60)

        iso   = IsolationForest(contamination=0.1, random_state=self.random_state)
        i_pr  = iso.fit_predict(self.X_train)
        print(f"Isolation Forest : {np.sum(i_pr==-1)} outliers "
              f"({np.sum(i_pr==-1)/len(i_pr)*100:.2f}%)")

        lof   = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        l_pr  = lof.fit_predict(self.X_train)
        print(f"LOF              : {np.sum(l_pr==-1)} outliers "
              f"({np.sum(l_pr==-1)/len(l_pr)*100:.2f}%)")
        print(f"ISO/LOF agreement: {np.mean(i_pr==l_pr):.2%}")

        for name, pr in [("ISO", i_pr), ("LOF", l_pr)]:
            fn = float(self.y_train[pr ==  1].mean())
            fa = float(self.y_train[pr == -1].mean()) if np.sum(pr == -1) > 0 else 0.0
            print(f"[{name}] Fail-rate normal: {fn:.2%} | anomalous: {fa:.2%} | "
                  f"{'✓ correlated' if fa > fn else '✗ weak'}")

    def detect_concept_drift(self, new_X, alpha: float = 0.05):
        print("\n" + "=" * 60)
        print("CONCEPT DRIFT DETECTION (KS-test + Bonferroni)")
        print("=" * 60)
        n   = min(self.X_train.shape[1], new_X.shape[1])
        adj = alpha / n
        print(f"Testing {n} features | Bonferroni α = {adj:.6f}")
        drifted = []
        for i in range(n):
            stat, p = ks_2samp(self.X_train.iloc[:, i], new_X.iloc[:, i])
            if p < adj:
                drifted.append((f"Feature_{i}", stat, p))
        if drifted:
            print(f"⚠  {len(drifted)} drifted features:")
            for feat, stat, p in drifted[:10]:
                print(f"   {feat}: KS={stat:.4f} p={p:.6f}")
        else:
            print("✓ No significant drift detected.")
        return drifted


# ═══════════════════════════════════════════════════════════════════════════════
#  MultiSeedRunner
# ═══════════════════════════════════════════════════════════════════════════════

class MultiSeedRunner:
    """
    Repeats the full pipeline across N random seeds and reports
    mean ± std for headline metrics — stronger than a single split claim.
    """

    def __init__(self, data_dir, output_dir, seeds=(0, 1, 2, 3, 4),
                 model_names=None, sampler_name="SMOTE",
                 feature_selection_method="rfe", use_robust_scaler=True):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seeds = list(seeds)
        self.model_names = model_names
        self.sampler_name = sampler_name
        self.feature_selection_method = feature_selection_method
        self.use_robust_scaler = use_robust_scaler
        self.per_seed: list = []

    def run(self) -> pd.DataFrame:
        for seed in self.seeds:
            print(f"\n{'#'*70}\n# SEED {seed}\n{'#'*70}")
            secom = AdvancedSECOM(
                data_dir=self.data_dir,
                output_dir=self.output_dir / f"seed_{seed}",
                random_state=seed)
            if not secom.load_data():
                continue
            secom.split_data(random_state=seed)
            secom.advanced_preprocessing(
                use_knn_imputer=False, use_robust_scaler=self.use_robust_scaler,
                feature_selection_method=self.feature_selection_method)

            all_m = secom.get_advanced_models()
            run_m = ({k: v for k, v in all_m.items() if k in self.model_names}
                     if self.model_names else all_m)
            sampler = secom.get_advanced_samplers().get(self.sampler_name)
            X_res, y_res = secom.X_train.copy(), secom.y_train.copy()
            if sampler is not None:
                X_res, y_res = sampler.fit_resample(X_res, y_res)

            for mname, model in run_m.items():
                try:
                    secom._evaluate_and_store(
                        key=f"{self.sampler_name}_{mname}", model=model,
                        model_name=mname, sampling_method=self.sampler_name,
                        X_resampled=X_res, y_resampled=y_res, sampler=sampler)
                except Exception as e:
                    print(f"  ✗ {mname} seed {seed}: {e}")

            self.per_seed.append(pd.DataFrame([
                {"seed": seed, "model": v["model_name"],
                 "f1": v["f1_score"], "mcc": v["mcc"],
                 "roc_auc": v["roc_auc"], "pr_auc": v["pr_auc"],
                 "precision": v["precision"], "recall": v["recall"]}
                for v in secom.results.values()
                if not v.get("is_degenerate", False)
            ]))
        return self.aggregate()

    def aggregate(self) -> pd.DataFrame:
        if not self.per_seed:
            return pd.DataFrame()
        full = pd.concat(self.per_seed, ignore_index=True)
        agg = full.groupby("model").agg(
            f1_mean=("f1", "mean"), f1_std=("f1", "std"),
            mcc_mean=("mcc", "mean"), mcc_std=("mcc", "std"),
            pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
            n_seeds=("seed", "nunique"),
        ).sort_values("f1_mean", ascending=False)
        print("\n" + "=" * 80)
        print(f"MULTI-SEED RESULTS (n={len(self.seeds)} seeds)")
        print("=" * 80)
        print(agg.to_string())
        agg.to_csv(self.output_dir / "multiseed_agg.csv")
        full.to_csv(self.output_dir / "multiseed_raw.csv", index=False)
        return agg


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry Point (equivalent to main.py)
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Advanced SECOM Predictive Maintenance System")
    print("=" * 55)

    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()

    secom = AdvancedSECOM(
        data_dir=script_dir / "secom",
        output_dir=script_dir / "output",
        random_state=42,
    )

    if not secom.load_data():
        print("Failed to load data."); return

    secom.split_data()
    secom.advanced_preprocessing(
        use_knn_imputer=False, use_robust_scaler=True,
        feature_selection_method="rfe")

    secom.train_advanced_models()
    secom.hyperparameter_optimization(top_models=3)

    secom.show_comprehensive_results()
    secom.show_detailed_classification_reports()
    secom.analyze_class_imbalance_impact()

    secom.feature_importance_analysis()
    secom.shap_analysis()
    secom.lime_analysis()
    secom.plot_learning_curves()
    secom.anomaly_detection_analysis()
    secom.detect_concept_drift(new_X=secom.X_test)

    best_key, best_result = secom.generate_final_report()
    print("\n✅ Analysis complete.")

    # ── Optional multi-seed robustness check ──────────────────────────────
    run_multiseed = True  # set True to produce Table with mean ± std
    if run_multiseed:
        runner = MultiSeedRunner(
            data_dir=secom.data_dir,
            output_dir=script_dir / "output" / "multiseed",
            seeds=range(10),
            model_names=["Balanced Random Forest", "XGBoost",
                         "CatBoost", "LightGBM"],
            sampler_name="SMOTE")
        runner.run()

    return secom, best_key, best_result


if __name__ == "__main__":
    result = main()
    if result:
        secom_system, best_model_key, best_model_result = result

