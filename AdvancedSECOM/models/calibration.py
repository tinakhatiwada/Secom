"""
AdvancedSECOM/models/calibration.py
CalibrationMixin — threshold optimisation and probability calibration.
"""
from __future__ import annotations

from ..dependencies import (
    np, f1_score,
    CalibratedClassifierCV, train_test_split,
)


class CalibrationMixin:
    """Mixin: threshold search and isotonic calibration."""

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
        """
        Leakage-free probability calibration using held-out calibration set.
        """
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

        # AUDIT FIX (F8): cv="prefit" was removed in scikit-learn >= 1.6, so the
        # original code silently caught the error and returned the UNCALIBRATED
        # model while the paper claimed calibration. Use FrozenEstimator on new
        # sklearn, fall back to cv="prefit" on old sklearn, and warn (never
        # silently) if calibration genuinely fails.
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
