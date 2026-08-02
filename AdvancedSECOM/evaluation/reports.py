"""
AdvancedSECOM/evaluation/reports.py
ReportsMixin — summary tables, classification reports, and final report.
"""
from __future__ import annotations

from ..dependencies import (
    np, pd, joblib,
    classification_report, confusion_matrix,
)


class ReportsMixin:
    """Mixin: result summaries, classification reports, cost analysis."""

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
