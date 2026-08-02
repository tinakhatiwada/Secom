"""
AdvancedSECOM/models/registry.py
ModelRegistryMixin — model and sampler factory dictionaries.
"""
from __future__ import annotations

from ..dependencies import (
    LogisticRegression, DecisionTreeClassifier,
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    SVC, MLPClassifier, KNeighborsClassifier, GaussianNB,
    BalancedBaggingClassifier, BalancedRandomForestClassifier,
    SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN,
    SMOTETomek, SMOTEENN, RandomUnderSampler,
    XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, CATBOOST_AVAILABLE,
)

# Conditional imports
if XGBOOST_AVAILABLE:
    from ..dependencies import xgb
if LIGHTGBM_AVAILABLE:
    from ..dependencies import lgb
if CATBOOST_AVAILABLE:
    from ..dependencies import cb


class ModelRegistryMixin:
    """Mixin: model and sampler factory methods."""

    def get_advanced_models(self) -> dict:
        models = {
            "Baseline LR": LogisticRegression(
                random_state=self.random_state, max_iter=1000,
                class_weight="balanced", penalty="l2", solver="lbfgs"),
            "Baseline Decision Tree": DecisionTreeClassifier(
                random_state=self.random_state, class_weight="balanced", max_depth=10),
            "Balanced Random Forest": BalancedRandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=self.random_state),
            "Balanced Bagging": BalancedBaggingClassifier(
                n_estimators=100, random_state=self.random_state),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=200, max_depth=15, random_state=self.random_state,
                class_weight="balanced"),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=8,
                random_state=self.random_state, subsample=0.8, min_samples_leaf=5),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=100, learning_rate=0.8, random_state=self.random_state,
                algorithm="SAMME"),
            "SVM RBF": SVC(
                kernel="rbf", probability=True, random_state=self.random_state,
                class_weight="balanced"),
            "Neural Network": MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500,
                random_state=self.random_state,
                early_stopping=True, validation_fraction=0.1),
            "Logistic Regression": LogisticRegression(
                random_state=self.random_state, max_iter=1000,
                class_weight="balanced", penalty="elasticnet",
                l1_ratio=0.5, solver="saga"),
            "K-Neighbors": KNeighborsClassifier(n_neighbors=7, weights="distance"),
            "Gaussian NB": GaussianNB(),
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                scale_pos_weight=float(self.class_weight_ratio), eval_metric="logloss",
                random_state=self.random_state, subsample=0.8, colsample_bytree=0.8)
        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=6, num_leaves=31,
                scale_pos_weight=float(self.class_weight_ratio),
                random_state=self.random_state, subsample=0.8, colsample_bytree=0.8,
                verbosity=-1)
        if CATBOOST_AVAILABLE:
            models["CatBoost"] = cb.CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=6,
                scale_pos_weight=float(self.class_weight_ratio),
                random_seed=self.random_state, verbose=False,
                eval_metric="F1", loss_function="Logloss")
        return models

    def get_advanced_samplers(self) -> dict:
        return {
            "Original (cost-sensitive)": None,
            "SMOTE":                SMOTE(random_state=self.random_state, k_neighbors=3),
            "Borderline SMOTE":     BorderlineSMOTE(random_state=self.random_state,
                                                    kind="borderline-1"),
            "SVM SMOTE":            SVMSMOTE(random_state=self.random_state),
            "ADASYN":               ADASYN(random_state=self.random_state),
            "SMOTE + Tomek":        SMOTETomek(random_state=self.random_state),
            "SMOTE + ENN":          SMOTEENN(random_state=self.random_state),
            "Random Undersampling": RandomUnderSampler(random_state=self.random_state,
                                                       sampling_strategy=0.3),
        }
