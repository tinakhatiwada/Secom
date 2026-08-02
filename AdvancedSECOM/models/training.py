"""
AdvancedSECOM/models/training.py
TrainingMixin — evaluate & store, training loop, ensembles.
"""
from __future__ import annotations

from ..dependencies import (
    traceback, time, np, pd,
    clone,
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    StratifiedKFold, cross_val_score,
    LogisticRegression, StackingClassifier, VotingClassifier,
    ImbPipeline,
)


class TrainingMixin:
    """Mixin: model evaluation, training loop, and ensemble creation."""

    def _evaluate_and_store(self, key, model, model_name, sampling_method,
                             X_resampled, y_resampled, sampler=None):
        model = clone(model)
        model.fit(X_resampled, y_resampled)

        # Calibrate + pick threshold using the held-out CAL slice, never X_test
        predict_model, best_thr = self._calibrate_and_threshold(model, self.X_cal, self.y_cal)

        # Inference latency (median of 10 single-sample runs)
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

        # Degenerate guard
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

        # Leakage-free CV (sampler inside pipeline)
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

        # Ensembles on best sampler only (to save time)
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
