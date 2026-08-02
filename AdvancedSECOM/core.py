"""
AdvancedSECOM/core.py
Composes all mixins into the final AdvancedSECOM class.
"""
from __future__ import annotations

from pathlib import Path

from .utils.helpers import fmt
from .data.preprocessing import DataLoaderMixin
from .models.registry import ModelRegistryMixin
from .models.calibration import CalibrationMixin
from .models.training import TrainingMixin
from .models.tuning import TuningMixin
from .evaluation.metrics import MetricsMixin
from .evaluation.reports import ReportsMixin
from .evaluation.plots import PlotsMixin
from .evaluation.explainability import ExplainabilityMixin
from .evaluation.diagnostics import DiagnosticsMixin


class AdvancedSECOM(
    DataLoaderMixin,
    ModelRegistryMixin,
    CalibrationMixin,
    TrainingMixin,
    TuningMixin,
    MetricsMixin,
    ReportsMixin,
    PlotsMixin,
    ExplainabilityMixin,
    DiagnosticsMixin,
):
    """Publication-ready SECOM predictive maintenance pipeline."""

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
