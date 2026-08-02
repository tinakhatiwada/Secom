"""
AdvancedSECOM/evaluation/explainability.py
ExplainabilityMixin — SHAP and LIME analysis.
"""
from __future__ import annotations

from ..dependencies import (
    traceback, np, plt,
    SHAP_AVAILABLE, LIME_AVAILABLE,
)

if SHAP_AVAILABLE:
    from ..dependencies import shap
if LIME_AVAILABLE:
    from ..dependencies import LimeTabularExplainer


class ExplainabilityMixin:
    """Mixin: SHAP and LIME model explanations."""

    def shap_analysis(self, max_display: int = 20):
        if not SHAP_AVAILABLE:
            print("SHAP not installed — pip install shap"); return

        nd = [(k, v) for k, v in self.results.items()
              if not v.get("is_degenerate", False)]
        if not nd:
            print("No non-degenerate models for SHAP."); return
        best_key, best_res = sorted(nd, key=lambda x: x[1]["f1_score"], reverse=True)[0]

        # Prefer the underlying base model (not CalibratedClassifierCV wrapper)
        # so TreeExplainer can introspect the tree structure
        model      = best_res["model"]
        model_name = best_res["model_name"]
        print(f"\nRunning SHAP on: {model_name}")

        X_arr = (self.X_test.values
                 if hasattr(self.X_test, "values") else np.asarray(self.X_test))
        X_shap = X_arr  # track which slice SHAP was actually run on

        try:
            # Try TreeExplainer first (fast, exact for tree models)
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

            # Binary → take class 1 slice
            if isinstance(shap_values, list):
                sv = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                sv = shap_values[:, :, 1]
            else:
                sv = shap_values

            # Summary (beeswarm)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(sv, X_shap, feature_names=self.feature_names,
                              max_display=max_display, show=False)
            plt.title(f"SHAP Summary — {model_name}")
            plt.tight_layout()
            plt.savefig(self.output_dir / "secom_shap_summary.png",
                        dpi=150, bbox_inches="tight")
            plt.show()

            # Bar (mean |SHAP|)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(sv, X_shap, feature_names=self.feature_names,
                              plot_type="bar", max_display=max_display, show=False)
            plt.title(f"SHAP Mean |Value| — {model_name}")
            plt.tight_layout()
            plt.savefig(self.output_dir / "secom_shap_bar.png",
                        dpi=150, bbox_inches="tight")
            plt.show()

            # Waterfall for sample 0
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
