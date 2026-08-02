"""
AdvancedSECOM/evaluation/metrics.py
MetricsMixin — bootstrap confidence intervals and McNemar's test.
"""
from __future__ import annotations

from ..dependencies import (
    np, pd,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    SCIPY_AVAILABLE, STATSMODELS_AVAILABLE,
)

if SCIPY_AVAILABLE:
    from ..dependencies import _chi2
if STATSMODELS_AVAILABLE:
    from ..dependencies import _sm_mcnemar


class MetricsMixin:
    """Mixin: bootstrap CIs and McNemar statistical comparison."""

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
        n10 = int(np.sum(corr_a & ~corr_b))   # A right, B wrong
        n01 = int(np.sum(~corr_a & corr_b))   # B right, A wrong
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
