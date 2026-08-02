"""
AdvancedSECOM/evaluation/diagnostics.py
DiagnosticsMixin — anomaly detection and concept drift.
"""
from __future__ import annotations

from ..dependencies import (
    np,
    IsolationForest, LocalOutlierFactor,
)


class DiagnosticsMixin:
    """Mixin: anomaly detection and concept drift detection."""

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
        from scipy.stats import ks_2samp
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
