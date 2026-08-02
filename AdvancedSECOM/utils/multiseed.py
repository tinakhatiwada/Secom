"""
AdvancedSECOM/utils/multiseed.py
Multi-seed robustness runner — repeats the full pipeline across N seeds.
"""
from __future__ import annotations

from pathlib import Path

from ..dependencies import (
    np, pd, clone,
    SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN,
    SMOTETomek, SMOTEENN, RandomUnderSampler,
)


class MultiSeedRunner:
    """
    Repeats the full pipeline across N random seeds and reports
    mean ± std for headline metrics — stronger than a single split claim.
    """

    def __init__(self, data_dir, output_dir, seeds=(0, 1, 2, 3, 4),
                 model_names=None, sampler_name="SMOTE",
                 feature_selection_method="rfe", use_robust_scaler=True):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seeds = list(seeds)
        self.model_names = model_names
        self.sampler_name = sampler_name
        self.feature_selection_method = feature_selection_method
        self.use_robust_scaler = use_robust_scaler
        self.per_seed: list = []

    def run(self) -> pd.DataFrame:
        # Import here to avoid circular import at module level
        from ..core import AdvancedSECOM

        for seed in self.seeds:
            print(f"\n{'#'*70}\n# SEED {seed}\n{'#'*70}")
            secom = AdvancedSECOM(
                data_dir=self.data_dir,
                output_dir=self.output_dir / f"seed_{seed}",
                random_state=seed)
            if not secom.load_data():
                continue
            secom.split_data(random_state=seed)
            secom.advanced_preprocessing(
                use_knn_imputer=False, use_robust_scaler=self.use_robust_scaler,
                feature_selection_method=self.feature_selection_method)

            all_m = secom.get_advanced_models()
            run_m = ({k: v for k, v in all_m.items() if k in self.model_names}
                     if self.model_names else all_m)
            sampler = secom.get_advanced_samplers().get(self.sampler_name)
            X_res, y_res = secom.X_train.copy(), secom.y_train.copy()
            if sampler is not None:
                X_res, y_res = sampler.fit_resample(X_res, y_res)

            for mname, model in run_m.items():
                try:
                    secom._evaluate_and_store(
                        key=f"{self.sampler_name}_{mname}", model=model,
                        model_name=mname, sampling_method=self.sampler_name,
                        X_resampled=X_res, y_resampled=y_res, sampler=sampler)
                except Exception as e:
                    print(f"  ✗ {mname} seed {seed}: {e}")

            self.per_seed.append(pd.DataFrame([
                {"seed": seed, "model": v["model_name"],
                 "f1": v["f1_score"], "mcc": v["mcc"],
                 "roc_auc": v["roc_auc"], "pr_auc": v["pr_auc"],
                 "precision": v["precision"], "recall": v["recall"]}
                for v in secom.results.values()
                if not v.get("is_degenerate", False)
            ]))
        return self.aggregate()

    def aggregate(self) -> pd.DataFrame:
        if not self.per_seed:
            return pd.DataFrame()
        full = pd.concat(self.per_seed, ignore_index=True)
        agg = full.groupby("model").agg(
            f1_mean=("f1", "mean"), f1_std=("f1", "std"),
            mcc_mean=("mcc", "mean"), mcc_std=("mcc", "std"),
            pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
            n_seeds=("seed", "nunique"),
        ).sort_values("f1_mean", ascending=False)
        print("\n" + "=" * 80)
        print(f"MULTI-SEED RESULTS (n={len(self.seeds)} seeds)")
        print("=" * 80)
        print(agg.to_string())
        agg.to_csv(self.output_dir / "multiseed_agg.csv")
        full.to_csv(self.output_dir / "multiseed_raw.csv", index=False)
        return agg
