"""
Main Execution Pipeline - Credit Risk Scorecard Project
------------------------------------------------------
Orchestrates data loading, absolute split isolation, two-tier preprocessing
safeguards, dynamic WOE transformation, dual-branch feature selection,
champion model auto-selection, multi-population artifact packages, and cloud sync.
"""

from dotenv import load_dotenv

load_dotenv()  # Auto-loads environment variables from .env file
import os
import time
import yaml
import joblib
import json
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from huggingface_hub import HfApi

# Import localized modules from the src factory
from src.data_loader import load_raw_training_data, split_train_val_test
from src.preprocessing import CreditDataCleaner, WOETransformer
from src.feature_selection import CreditFeatureSelector
from src.model import CreditModelTrainer
from src.scorecard import CreditScorecardScaler
from src.config import ARTIFACTS_DIR
from src.utils import (
    get_run_directory,
    save_roc_curve,
    save_classification_report,
    save_iv_scores,
    save_woe_tables,
    save_probability_distribution,
    save_confusion_matrix_heatmap,
    log_progress,
    extract_structural_bins,
    evaluate_and_select_champion,
    calculate_calibration_error,
)


def package_and_upload_artifacts(run_dir: Path, config: Dict[str, Any]) -> None:
    """
    Packages production model binaries into model.zip and syncs with Hugging Face Hub.
    """
    models_dir = run_dir / "models"
    zip_path = run_dir / "model.zip"

    print("\n" + "=" * 70)
    print("📦 [MLOPS CLOUD] PACKAGING MODEL ARTIFACTS FOR DEPLOYMENT")
    print("=" * 70)

    # 1. Zip model binaries
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in models_dir.glob("*"):
            if file_path.is_file():
                zipf.write(file_path, arcname=file_path.name)
    print(f"  -> Successfully zipped production binaries to: {zip_path.name}")

    # 2. Upload to Hugging Face Hub if token available
    hf_token = os.getenv("HF_TOKEN")
    repo_id = "trungcan94/AIO_moddule_2_model"

    if hf_token:
        try:
            print(f"  -> Uploading 'model.zip' to Hugging Face Dataset: {repo_id}...")
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(zip_path),
                path_in_repo="model.zip",
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
            )
            print("  ✅ [SUCCESS] Production model published to Hugging Face Cloud!")
        except Exception as e:
            print(f"  ⚠️ [UPLOAD WARNING] Cloud sync failed: {e}")
    else:
        print("  ℹ️ [LOCAL SYNC] HF_TOKEN environment variable not detected.")
        print(
            "     Local 'model.zip' created. Streamlit inference engine will read locally."
        )


def main() -> None:
    """
    Master orchestrator function for the Credit Risk Scorecard development pipeline.
    """
    print("======================================================================")
    print("🚀 STARTING PRODUCTION-GRADE CREDIT RISK SCORECARD WORKFLOW")
    print("======================================================================")

    start_time: float = time.time()
    TOTAL_STEPS: int = 8

    # --------------------------------------------------------------------------
    # STEP 1: LOAD SYSTEM CONFIGURATION
    # --------------------------------------------------------------------------
    log_progress(
        1, TOTAL_STEPS, "Loading Configuration Registry (config.yaml)", start_time
    )
    config_path: str = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"[CRITICAL] Configuration file missing at: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    NUM_COLS: List[str] = config["features"]["numerical"]
    CAT_COLS: List[str] = config["features"]["categorical"]
    TARGET_COL: str = config["target"]

    # --------------------------------------------------------------------------
    # STEP 2: RAW DATA ACQUISITION
    # --------------------------------------------------------------------------
    log_progress(2, TOTAL_STEPS, "Acquiring Raw Data Registry from Disk", start_time)
    data_path: str = os.path.join("data", "raw", config["data"]["raw_file_name"])
    df_raw: pd.DataFrame = load_raw_training_data(file_path=data_path)

    # --------------------------------------------------------------------------
    # STEP 3: ANTI-LEAKAGE STRATIFIED PARTITIONING & TARGET ISOLATION
    # --------------------------------------------------------------------------
    log_progress(
        3, TOTAL_STEPS, "Executing Stratified Data Split Segregation", start_time
    )

    print("Validating Target Column ...")
    if df_raw[TARGET_COL].isnull().any():
        raise ValueError(
            f"[DQ-01 FAILED] Target column '{TARGET_COL}' contains Null values."
        )

    unique_targets = set(df_raw[TARGET_COL].dropna().unique())
    if unique_targets != {0, 1}:
        raise ValueError(
            f"[DQ-01 FAILED] Target column must be binary {{0, 1}}. Found: {unique_targets}"
        )
    print("  -> Target validation passed: Binary {0, 1} confirmed.")

    split_params: Dict[str, Any] = config["data"]["split_params"]

    df_train, df_val, df_test = split_train_val_test(
        df=df_raw, target_column=TARGET_COL, split_params=split_params
    )

    X_train: pd.DataFrame = df_train.drop(columns=[TARGET_COL]).copy()
    y_train: pd.Series = df_train[TARGET_COL].copy()

    X_val: pd.DataFrame = df_val.drop(columns=[TARGET_COL]).copy()
    y_val: pd.Series = df_val[TARGET_COL].copy()

    X_test: pd.DataFrame = df_test.drop(columns=[TARGET_COL]).copy()
    y_test: pd.Series = df_test[TARGET_COL].copy()

    # --------------------------------------------------------------------------
    # STEP 4: TIER-1 CLEANING PIPELINE (STATISTICAL & BUSINESS CONSTRAINTS)
    # --------------------------------------------------------------------------
    log_progress(4, TOTAL_STEPS, "Running Tier-1 Data Cleaner Pipeline", start_time)
    cleaner: CreditDataCleaner = CreditDataCleaner(
        numerical_features=NUM_COLS,
        categorical_features=CAT_COLS,
        cleaning_config=config.get("cleaning_config", {}),
    )

    global_mutation_log = []
    cleaning_summaries = {}

    # --- 4.1 TRAINING SET AUDIT ---
    print("\n" + "-" * 70)
    print("📊 [TIER-1 AUDIT] EVALUATING TRAINING SET")
    X_train_clean: pd.DataFrame = cleaner.fit_transform(X_train)
    cleaner._print_audit_log()

    train_mutations = cleaner.audit_report_.get("mutation_details", [])
    for mutation in train_mutations:
        mutation["dataset_type"] = "train"
    global_mutation_log.extend(train_mutations)

    cleaning_summaries["train"] = {
        "total_input_records": cleaner.audit_report_.get("total_input_records"),
        "final_records": cleaner.audit_report_.get("final_records"),
        "outliers_dropped_count": cleaner.audit_report_.get("outliers_dropped_count"),
        "outliers_dropped_ratio": cleaner.audit_report_.get("outliers_dropped_ratio"),
        "reason_codes_flagged": cleaner.audit_report_.get("reason_codes_flagged"),
        "missing_values_imputed": cleaner.audit_report_.get("missing_values_imputed"),
    }
    y_train = y_train.loc[X_train_clean.index]

    # --- 4.2 VALIDATION SET AUDIT ---
    print("\n" + "-" * 70)
    print("📊 [TIER-1 AUDIT] EVALUATING VALIDATION SET")
    X_val_clean: pd.DataFrame = cleaner.transform(X_val)
    cleaner._print_audit_log()

    val_mutations = cleaner.audit_report_.get("mutation_details", [])
    for mutation in val_mutations:
        mutation["dataset_type"] = "validation"
    global_mutation_log.extend(val_mutations)

    cleaning_summaries["validation"] = {
        "total_input_records": cleaner.audit_report_.get("total_input_records"),
        "final_records": cleaner.audit_report_.get("final_records"),
        "outliers_dropped_count": cleaner.audit_report_.get("outliers_dropped_count"),
        "outliers_dropped_ratio": cleaner.audit_report_.get("outliers_dropped_ratio"),
        "reason_codes_flagged": cleaner.audit_report_.get("reason_codes_flagged"),
        "missing_values_imputed": cleaner.audit_report_.get("missing_values_imputed"),
    }
    y_val = y_val.loc[X_val_clean.index]

    # --- 4.3 TESTING SET AUDIT ---
    print("\n" + "-" * 70)
    print("📊 [TIER-1 AUDIT] EVALUATING TESTING SET")
    X_test_clean: pd.DataFrame = cleaner.transform(X_test)
    cleaner._print_audit_log()

    test_mutations = cleaner.audit_report_.get("mutation_details", [])
    for mutation in test_mutations:
        mutation["dataset_type"] = "test"
    global_mutation_log.extend(test_mutations)

    cleaning_summaries["test"] = {
        "total_input_records": cleaner.audit_report_.get("total_input_records"),
        "final_records": cleaner.audit_report_.get("final_records"),
        "outliers_dropped_count": cleaner.audit_report_.get("outliers_dropped_count"),
        "outliers_dropped_ratio": cleaner.audit_report_.get("outliers_dropped_ratio"),
        "reason_codes_flagged": cleaner.audit_report_.get("reason_codes_flagged"),
        "missing_values_imputed": cleaner.audit_report_.get("missing_values_imputed"),
    }
    y_test = y_test.loc[X_test_clean.index]

    cleaner.audit_report_["mutation_details"] = global_mutation_log

    # --------------------------------------------------------------------------
    # STEP 5: TIER-2 DYNAMIC WOE BINNING & ENCODING
    # --------------------------------------------------------------------------
    log_progress(5, TOTAL_STEPS, "Running Tier-2 WOE Transformation", start_time)
    bin_config: Dict[str, Any] = config["features"]["bin_config"]
    diagnostics_config: Dict[str, float] = config["features"].get("woe_diagnostics", {})

    woe_transformer: WOETransformer = WOETransformer(
        numerical_features=NUM_COLS,
        categorical_features=CAT_COLS,
        bin_config=bin_config,
        diagnostics_config=diagnostics_config,
    )

    X_train_woe: pd.DataFrame = woe_transformer.fit_transform(X_train_clean, y_train)
    X_val_woe: pd.DataFrame = woe_transformer.transform(X_val_clean)
    X_test_woe: pd.DataFrame = woe_transformer.transform(X_test_clean)

    # --------------------------------------------------------------------------
    # STEP 6: TIER-3 DUAL-BRANCH FEATURE SELECTION
    # --------------------------------------------------------------------------
    log_progress(
        6, TOTAL_STEPS, "Running Tier-3 Dual-Branch Feature Selection", start_time
    )

    selection_config = config["features"].get("selection_config", {})
    feature_selector = CreditFeatureSelector(selection_config=selection_config)

    X_train_sel_dict: Dict[str, pd.DataFrame] = feature_selector.fit_transform(
        X_train_woe, y_train, woe_transformer.iv_scores
    )

    X_val_sel_dict: Dict[str, pd.DataFrame] = feature_selector.transform(X_val_woe)
    X_test_sel_dict: Dict[str, pd.DataFrame] = feature_selector.transform(X_test_woe)

    # --------------------------------------------------------------------------
    # STEP 7: DUAL MODEL TRAINING & CHAMPION AUTO-SELECTION
    # --------------------------------------------------------------------------
    log_progress(
        7, TOTAL_STEPS, "Training Competitors & Selecting Champion Model", start_time
    )

    model_baseline = CreditModelTrainer(model_config=config["model"])
    model_subset = CreditModelTrainer(model_config=config["model"])

    model_baseline.fit(X_train_sel_dict["baseline"], y_train)
    model_subset.fit(X_train_sel_dict["subset"], y_train)

    y_val_prob_base = model_baseline.predict_probability(X_val_sel_dict["baseline"])
    y_val_prob_sub = model_subset.predict_probability(X_val_sel_dict["subset"])
    y_val_true_arr = y_val.to_numpy()

    champ_cfg = selection_config.get("champion_selection", {})
    champion_key, champion_model, metrics_log = evaluate_and_select_champion(
        y_val_true=y_val_true_arr,
        y_val_prob_base=y_val_prob_base,
        y_val_prob_sub=y_val_prob_sub,
        model_baseline=model_baseline,
        model_subset=model_subset,
        champ_cfg=champ_cfg,
    )

    # --------------------------------------------------------------------------
    # STEP 8: MLOPS ARTIFACT STORAGE PACKAGING & MULTI-POPULATION EXPORT
    # --------------------------------------------------------------------------
    log_progress(
        8, TOTAL_STEPS, "Generating Isolated Run Packages & Visualizations", start_time
    )

    current_run_dir: Path = get_run_directory(ARTIFACTS_DIR)
    print(f"[MLOPS] Storage Target Directory Activated: {current_run_dir.name}")

    for sub_dir in ["plots", "metrics", "models", "data", "tables"]:
        (current_run_dir / sub_dir).mkdir(parents=True, exist_ok=True)

    # --- FINANCIAL SCORE SCALE CONVERSION ---
    print(f"[MLOPS] Activating Scorecard Scaling for Champion ({champion_key})...")
    score_scaler = CreditScorecardScaler(scaling_config=config["scorecard_scaling"])
    score_scaler.fit(model_trainer=champion_model, woe_transformer=woe_transformer)

    X_train_bins = extract_structural_bins(X_train_clean, woe_transformer)
    X_val_bins = extract_structural_bins(X_val_clean, woe_transformer)
    X_test_bins = extract_structural_bins(X_test_clean, woe_transformer)

    train_scores: pd.Series = score_scaler.transform(X_train_bins)
    val_scores: pd.Series = score_scaler.transform(X_val_bins)
    test_scores: pd.Series = score_scaler.transform(X_test_bins)

    # --- FINANCIAL CALIBRATION AUDIT ---
    y_val_prob_champ = y_val_prob_base if champion_key == "baseline" else y_val_prob_sub
    calibration_metrics = calculate_calibration_error(
        y_prob=y_val_prob_champ,
        actual_scores=val_scores,
        scaling_config=config["scorecard_scaling"],
    )

    with open(
        current_run_dir / "metrics" / "pdo_calibration_error.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(calibration_metrics, f, indent=4)

    # --- PHYSICAL DATASETS EXPORTATION ---
    print("[MLOPS] Saving processed row-level datasets to storage records...")

    train_historical = X_train_sel_dict[champion_key].copy()
    train_historical["credit_score"] = train_scores
    train_historical[TARGET_COL] = y_train

    val_historical = X_val_sel_dict[champion_key].copy()
    val_historical["credit_score"] = val_scores
    val_historical[TARGET_COL] = y_val

    test_historical = X_test_sel_dict[champion_key].copy()
    test_historical["credit_score"] = test_scores
    test_historical[TARGET_COL] = y_test

    train_historical.to_csv(
        current_run_dir / "data" / f"train_woe_{champion_key}.csv", index=False
    )
    val_historical.to_csv(
        current_run_dir / "data" / f"val_woe_{champion_key}.csv", index=False
    )
    test_historical.to_csv(
        current_run_dir / "data" / f"test_woe_{champion_key}.csv", index=False
    )

    # --- AUDIT LOGS EXPORT ---
    print("[MLOPS] Saving Validation & Quality Audit Logs...")

    dropped_outliers_df = df_train.loc[~df_train.index.isin(X_train_clean.index)].copy()
    if not dropped_outliers_df.empty:
        dropped_outliers_df.to_csv(
            current_run_dir / "tables" / "qa_dropped_outliers_audit.csv",
            index=True,
            index_label="original_index",
        )

    with open(
        current_run_dir / "tables" / "data_cleaning_summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(cleaning_summaries, f, indent=4)

    if cleaner.audit_report_.get("mutation_details"):
        pd.DataFrame(cleaner.audit_report_["mutation_details"]).to_csv(
            current_run_dir / "tables" / "data_mutation_audit_log.csv", index=False
        )

    if woe_transformer.diagnostic_warnings_:
        pd.DataFrame(woe_transformer.diagnostic_warnings_).to_csv(
            current_run_dir / "tables" / "woe_diagnostics_warnings.csv", index=False
        )

    with open(
        current_run_dir / "metrics" / "champion_vs_challenger_comparison.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics_log, f, indent=4)

    with open(
        current_run_dir / "metrics" / "feature_selection_audit.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(feature_selector.audit_report_, f, indent=4)

    save_iv_scores(
        woe_transformer.iv_scores,
        current_run_dir / "metrics" / "baseline_iv_scores.json",
    )
    save_woe_tables(woe_transformer.woe_dictionaries, current_run_dir / "tables")

    score_scaler.export_artifacts(folder_path=current_run_dir / "tables")
    score_scaler.plot_monotonic_barcharts(folder_path=current_run_dir / "plots")

    # --- POPULATION EVALUATIONS ---
    X_train_champ = X_train_sel_dict[champion_key]
    X_val_champ = X_val_sel_dict[champion_key]
    X_test_champ = X_test_sel_dict[champion_key]

    y_train_pred: np.ndarray = champion_model.predict_class(X_train_champ)
    y_train_prob: np.ndarray = champion_model.predict_probability(X_train_champ)
    save_classification_report(
        y_true=y_train,
        y_pred=y_train_pred,
        y_prob=y_train_prob,
        file_path=current_run_dir / "metrics" / "train_evaluation_report.json",
    )

    y_val_pred: np.ndarray = champion_model.predict_class(X_val_champ)
    y_val_prob: np.ndarray = champion_model.predict_probability(X_val_champ)
    save_classification_report(
        y_true=y_val,
        y_pred=y_val_pred,
        y_prob=y_val_prob,
        file_path=current_run_dir / "metrics" / "validation_evaluation_report.json",
    )
    save_roc_curve(
        y_true=y_val,
        y_prob=y_val_prob,
        file_path=current_run_dir / "plots" / "validation_roc_curve.png",
    )
    save_probability_distribution(
        y_true=y_val,
        y_prob=y_val_prob,
        file_path=current_run_dir / "plots" / "validation_probability_distribution.png",
    )
    save_confusion_matrix_heatmap(
        y_true=y_val,
        y_pred=y_val_pred,
        file_path=current_run_dir / "plots" / "validation_confusion_matrix_heatmap.png",
    )

    y_test_pred: np.ndarray = champion_model.predict_class(X_test_champ)
    y_test_prob: np.ndarray = champion_model.predict_probability(X_test_champ)
    save_classification_report(
        y_true=y_test,
        y_pred=y_test_pred,
        y_prob=y_test_prob,
        file_path=current_run_dir / "metrics" / "test_evaluation_report.json",
    )
    save_roc_curve(
        y_true=y_test,
        y_prob=y_test_prob,
        file_path=current_run_dir / "plots" / "test_roc_curve.png",
    )
    save_probability_distribution(
        y_true=y_test,
        y_prob=y_test_prob,
        file_path=current_run_dir / "plots" / "test_probability_distribution.png",
    )
    save_confusion_matrix_heatmap(
        y_true=y_test,
        y_pred=y_test_pred,
        file_path=current_run_dir / "plots" / "test_confusion_matrix_heatmap.png",
    )

    # --- MODEL BINARY LOCK ---
    model_name = config["model"]["logistic_regression"].get(
        "model_file_name", "champion_logistic_model.pkl"
    )
    joblib.dump(champion_model, current_run_dir / "models" / model_name)
    joblib.dump(cleaner, current_run_dir / "models" / "cleaner.pkl")
    joblib.dump(woe_transformer, current_run_dir / "models" / "woe_transformer.pkl")
    joblib.dump(feature_selector, current_run_dir / "models" / "feature_selector.pkl")
    joblib.dump(score_scaler, current_run_dir / "models" / "score_scaler.pkl")

    print("[MLOPS] Production artifacts successfully locked.")

    # --- AUTOMATED MODEL PACKAGING & CLOUD SYNC ---
    package_and_upload_artifacts(run_dir=current_run_dir, config=config)

    # ==============================================================================
    # FINAL EXECUTIVE PERFORMANCE AUDIT SUMMARIES
    # ==============================================================================
    print("\n======================================================================")
    print(
        f"📊 TRIPLE-POPULATION METRICS COMPLIANCE GRID (CHAMPION: {champion_key.upper()})"
    )
    print("======================================================================")
    with open(
        current_run_dir / "metrics" / "train_evaluation_report.json", "r"
    ) as trf, open(
        current_run_dir / "metrics" / "validation_evaluation_report.json", "r"
    ) as vf, open(
        current_run_dir / "metrics" / "test_evaluation_report.json", "r"
    ) as tf:
        tr_scores = json.load(trf)["scores"]
        v_scores = json.load(vf)["scores"]
        t_scores = json.load(tf)["scores"]

    print(f"  Metric Profile      | Train Dataset | Validation Set | Testing Dataset")
    print(f"  --------------------|---------------|----------------|----------------")
    print(
        f"  Kolmogorov-Smirnov  | {tr_scores['kolmogorov_smirnov_ks']:>13.4f} | {v_scores['kolmogorov_smirnov_ks']:>14.4f} | {t_scores['kolmogorov_smirnov_ks']:>15.4f}"
    )
    print(
        f"  F1-Classification   | {tr_scores['f1_score']:>13.4f} | {v_scores['f1_score']:>14.4f} | {t_scores['f1_score']:>15.4f}"
    )
    print(
        f"  Overall Accuracy    | {tr_scores['accuracy']:>13.4f} | {v_scores['accuracy']:>14.4f} | {t_scores['accuracy']:>15.4f}"
    )

    print("\n======================================================================")
    print("📋 REGULATORY RISK COEFFICIENTS VALIDATION AUDIT")
    print("======================================================================")
    print(f"Base Intercept (Beta_0) : {champion_model.intercept_:.4f}")

    all_betas_valid: bool = True
    for feature, beta in champion_model.coefficients_.items():
        is_valid = beta < 0
        status_icon = "✅ VALID" if is_valid else "❌ CRITICAL ERROR"
        if not is_valid:
            all_betas_valid = False
        print(f"  - {feature:<22} | Beta: {beta:.4f} ({status_icon})")

    print("-" * 70)
    if all_betas_valid:
        print("🛡️ RISK STATUS: PASSED. Model complies with credit risk mathematics.")
    else:
        print("🚨 RISK STATUS: FAILED. Review positive features for trend reversal.")
    print("======================================================================")

    # --- FINANCIAL CALIBRATION AUDIT SUMMARY ---
    print("\n======================================================================")
    print("⚖️ FINANCIAL CALIBRATION AUDIT (PDO ROUNDING ERROR)")
    print("======================================================================")
    print(
        f"  -> Mean Absolute Error (MAE) : {calibration_metrics['mean_absolute_error_points']:.4f} points"
    )
    print(
        f"  -> Maximum Point Deviation   : {calibration_metrics['max_absolute_error_points']:.4f} points"
    )
    print(
        f"  -> Calibration Variance      : {calibration_metrics['calibration_variance_ratio']:.4f} (Ideal is ~1.0)"
    )

    total_elapsed: float = time.time() - start_time
    print(
        f"\n[SUCCESS] End-to-End master pipeline executed cleanly in {total_elapsed:.2f} seconds!"
    )
    print(
        f"[MLOPS] All decoupled engineering packages are locked at: {current_run_dir}\n"
    )


if __name__ == "__main__":
    main()
