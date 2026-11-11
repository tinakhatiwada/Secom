"""
main.py — Entry point for the AdvancedSECOM pipeline.
Equivalent to the main() function previously in dm1.py.
"""
from pathlib import Path
from AdvancedSECOM import AdvancedSECOM


def main():
    print("Advanced SECOM Predictive Maintenance System")
    print("=" * 55)

    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path(r"C:\Users\sansk\Downloads\Secom\secom")

    secom = AdvancedSECOM(
        data_dir=Path(r"C:\Users\sansk\Downloads\Secom\secom"),
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

    # ── Optional multi-seed robustness / reviewer evidence ────────────────
    # Produces the multi-seed table (Balanced Bagging + cost-sensitive baseline
    # + all leading configs), per-seed thresholds, McNemar significance,
    # calibration diagnostics (Brier/ECE + reliability diagram) and the
    # chronological-split result. This is the SINGLE correct multi-seed path
    # (the old MultiSeedRunner, which ran the wrong models with no baseline,
    # thresholds or calibration, is no longer used). Seeds = 42 plus nine more,
    # matching the manuscript.
    run_evidence = True  # set True to regenerate all reviewer-facing tables/figures
    if run_evidence:
        from run_all import reproduce_all
        reproduce_all(
            data_dir=secom.data_dir,
            out_dir=script_dir / "output" / "evidence",
            seeds=[42, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    return secom, best_key, best_result


if __name__ == "__main__":
    result = main()
    if result:
        secom_system, best_model_key, best_model_result = result
