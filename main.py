"""
main.py — Entry point for the AdvancedSECOM pipeline.
Equivalent to the main() function previously in dm1.py.
"""
from pathlib import Path
from AdvancedSECOM import AdvancedSECOM, MultiSeedRunner


def main():
    print("Advanced SECOM Predictive Maintenance System")
    print("=" * 55)

    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path(r"C:\Users\sansk\Downloads\revisionAI\secom")

    secom = AdvancedSECOM(
        data_dir=Path(r"C:\Users\sansk\Downloads\revisionAI\secom"),
        output_dir=script_dir / "output",
        random_state=42,
    )

    if not secom.load_data():
        print("Failed to load data."); return

    secom.split_data()
    secom.advanced_preprocessing(
        use_knn_imputer=False, use_robust_scaler=True,
        feature_selection_method="rfe")

    secom.train_advanced_models()
    secom.hyperparameter_optimization(top_models=3)

    secom.show_comprehensive_results()
    secom.show_detailed_classification_reports()
    secom.analyze_class_imbalance_impact()

    secom.feature_importance_analysis()
    secom.shap_analysis()
    secom.lime_analysis()
    secom.plot_learning_curves()
    secom.anomaly_detection_analysis()
    secom.detect_concept_drift(new_X=secom.X_test)

    best_key, best_result = secom.generate_final_report()
    print("\n✅ Analysis complete.")

    # ── Optional multi-seed robustness check ──────────────────────────────
    run_multiseed = False   # set True to produce Table with mean ± std
    if run_multiseed:
        runner = MultiSeedRunner(
            data_dir=secom.data_dir,
            output_dir=script_dir / "output" / "multiseed",
            seeds=range(10),
            model_names=["Balanced Random Forest", "XGBoost",
                         "CatBoost", "LightGBM"],
            sampler_name="SMOTE")
        runner.run()

    return secom, best_key, best_result


if __name__ == "__main__":
    result = main()
    if result:
        secom_system, best_model_key, best_model_result = result
