"""
AdvancedSECOM/models/tuning.py
TuningMixin — hyperparameter optimisation via GridSearchCV.
"""
from __future__ import annotations

from ..dependencies import (
    traceback, pd, clone,
    GridSearchCV, StratifiedKFold, ImbPipeline,
)


class TuningMixin:
    """Mixin: hyperparameter optimisation for top models."""

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

            # AUDIT FIX (F9): the original code resampled the FULL training set
            # and THEN ran GridSearchCV, so SMOTE synthetic points leaked across
            # CV folds — inflating CV scores to ~1.0 (see audit B3). The sampler
            # must live INSIDE the cross-validation pipeline so each fold is
            # resampled from its own training portion only.
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
            search.fit(self.X_train, self.y_train)   # raw train — no pre-resampling

            # Rebuild the tuned base model, then evaluate it through the same
            # resample→fit→calibrate path used for every other configuration.
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
