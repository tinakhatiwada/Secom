"""
AdvancedSECOM/models/tuning.py
TuningMixin — hyperparameter optimisation via GridSearchCV.
"""
from __future__ import annotations

from ..dependencies import (
    traceback, pd, clone,
    GridSearchCV,
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
            X_res, y_res = self.X_train.copy(), self.y_train.copy()
            if sampler is not None:
                X_res, y_res = clone(sampler).fit_resample(X_res, y_res)

            search = GridSearchCV(
                self.get_advanced_models()[mn], param_grids[mn],
                cv=3, scoring="average_precision", n_jobs=-1, verbose=0)
            search.fit(X_res, y_res)

            opt_key = f"{key}_OPTIMIZED"
            try:
                self._evaluate_and_store(
                    key=opt_key, model=search.best_estimator_,
                    model_name=f"{mn} (Optimized)", sampling_method=sm,
                    X_resampled=X_res, y_resampled=y_res, sampler=sampler)
                self.results[opt_key]["best_params"] = search.best_params_
                print(f"  Best params : {search.best_params_}")
                print(f"  Opt F1      : {self.results[opt_key]['f1_score']:.4f} "
                      f"(orig: {res['f1_score']:.4f})")
            except Exception as e:
                print(f"  ✗ {e}"); traceback.print_exc()
