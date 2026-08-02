"""
AdvancedSECOM/evaluation/plots.py
PlotsMixin — performance visualisations, learning curves, feature importance.
"""
from __future__ import annotations

from ..dependencies import (
    np, pd, plt, sns,
    confusion_matrix, roc_curve, precision_recall_curve,
    StratifiedKFold, learning_curve,
)


class PlotsMixin:
    """Mixin: all matplotlib/seaborn visualisations."""

    def create_advanced_visualizations(self, df: pd.DataFrame):
        nd = df[~df["Degenerate"]]
        if nd.empty:
            print("No non-degenerate models to plot."); return

        fig, axes = plt.subplots(2, 3, figsize=(22, 12))

        # Top-10 F1 with CI error bars
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

        # Sampling avg F1
        samp = nd.groupby("Sampling")["F1"].mean().sort_values(ascending=False)
        axes[0, 1].bar(range(len(samp)), samp.values)
        axes[0, 1].set_xticks(range(len(samp)))
        axes[0, 1].set_xticklabels(samp.index, rotation=45, ha="right", fontsize=7)
        axes[0, 1].set_ylabel("Avg F1"); axes[0, 1].set_title("Sampling Methods")
        axes[0, 1].grid(True, alpha=0.3)

        # Precision-Recall scatter
        sc = axes[0, 2].scatter(nd["Precision"], nd["Recall"],
                                c=nd["F1"], cmap="viridis", alpha=0.7)
        plt.colorbar(sc, ax=axes[0, 2], label="F1")
        axes[0, 2].set_xlabel("Precision"); axes[0, 2].set_ylabel("Recall")
        axes[0, 2].set_title("Precision vs Recall"); axes[0, 2].grid(True, alpha=0.3)

        # Model avg F1
        mp = nd.groupby("Model")["F1"].mean().sort_values(ascending=False)
        axes[1, 0].bar(range(len(mp)), mp.values)
        axes[1, 0].set_xticks(range(len(mp)))
        axes[1, 0].set_xticklabels(mp.index, rotation=45, ha="right", fontsize=7)
        axes[1, 0].set_ylabel("Avg F1"); axes[1, 0].set_title("Model Performance")
        axes[1, 0].grid(True, alpha=0.3)

        # CV-vs-Test gap
        axes[1, 1].scatter(nd["CV_F1"], nd["F1"], alpha=0.6)
        lim = max(nd["F1"].max(), nd["CV_F1"].fillna(0).max()) + 0.05
        axes[1, 1].plot([0, lim], [0, lim], "k--", alpha=0.4, label="y=x")
        axes[1, 1].set_xlabel("CV F1 (leakage-free)"); axes[1, 1].set_ylabel("Test F1")
        axes[1, 1].set_title("CV vs Test F1 (gap)"); axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        # Latency
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

        # Confusion matrix of best model
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

        # ROC curves
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

        # PR curves
        plt.figure(figsize=(10, 8))
        for _, row in nd.head(5).iterrows():
            r = self.results[row["Key"]]
            if r["predictions_proba"] is not None:
                prec, rec, _ = precision_recall_curve(self.y_test, r["predictions_proba"])
                plt.plot(rec, prec,
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
