"""
Main Execution Pipeline - Credit Risk Scorecard Project
------------------------------------------------------
Orchestrates data loading, absolute split isolation, two-tier preprocessing
safeguards, dynamic WOE transformation, model training, training sanity checks,
and dual-population validation/testing artifact packages with real-time ETA progress logs.
"""

import os
import time
import yaml
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Import localized modules from the src factory
from src.data_loader import load_raw_training_data, split_train_val_test
from src.preprocessing import CreditDataCleaner, WOETransformer
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
)


def main() -> None:
    """
    Master orchestrator function for the Credit Risk Scorecard development pipeline.
    """
    print("======================================================================")
    print("🚀 STARTING PRODUCTION-GRADE CREDIT RISK SCORECARD WORKFLOW")
    print("======================================================================")

    start_time: float = time.time()
    TOTAL_STEPS: int = 7

    # --------------------------------------------------------------------------
    # STEP 1: LOAD SYSTEM CONFIGURATION
    # --------------------------------------------------------------------------
    # PURPOSE: Fetch runtime parameters centrally to eliminate hardcoded values.
    # INPUTS:  File system text path ("config.yaml").
    # OUTPUTS: List[str] NUM_COLS, List[str] CAT_COLS, str TARGET_COL, Dict config registry.
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
    # PURPOSE: Pull the primary structural data table from local storage into memory.
    # INPUTS:  Constructed relative OS directory path string ("data/raw/credit_risk_dataset.csv").
    # OUTPUTS: pd.DataFrame df_raw containing the original unmodified dataset matrix.
    # --------------------------------------------------------------------------
    log_progress(2, TOTAL_STEPS, "Acquiring Raw Data Registry from Disk", start_time)
    data_path: str = os.path.join("data", "raw", config["data"]["raw_file_name"])
    df_raw: pd.DataFrame = load_raw_training_data(file_path=data_path)

    # --------------------------------------------------------------------------
    # STEP 3: ANTI-LEAKAGE STRATIFIED PARTITIONING & TARGET ISOLATION
    # --------------------------------------------------------------------------
    # PURPOSE: Segregate data into independent evaluation populations and strip target variables.
    # INPUTS:  pd.DataFrame df_raw, str TARGET_COL, Dict split_params (sizes, random_state).
    # OUTPUTS: Feature matrices (X_train, X_val, X_test) and target arrays (y_train, y_val, y_test).
    # --------------------------------------------------------------------------
    log_progress(
        3, TOTAL_STEPS, "Executing Stratified Data Split Segregation", start_time
    )
    split_params: Dict[str, Any] = config["data"]["split_params"]

    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame
    df_train, df_val, df_test = split_train_val_test(
        df=df_raw, target_column=TARGET_COL, split_params=split_params
    )

    # TARGET ISOLATION STRATEGY: Strip away loan_status immediately to block leakage
    X_train: pd.DataFrame = df_train[NUM_COLS + CAT_COLS].copy()
    y_train: pd.Series = df_train[TARGET_COL].copy()

    X_val: pd.DataFrame = df_val[NUM_COLS + CAT_COLS].copy()
    y_val: pd.Series = df_val[TARGET_COL].copy()

    X_test: pd.DataFrame = df_test[NUM_COLS + CAT_COLS].copy()
    y_test: pd.Series = df_test[TARGET_COL].copy()

    # --------------------------------------------------------------------------
    # STEP 4: TIER-1 CLEANING PIPELINE ISO-CONSTRAINTS
    # --------------------------------------------------------------------------
    # PURPOSE: Handle null entries by imputing uniform risk flags (-1.0 and "Missing").
    # INPUTS:  Feature dataframes (X_train, X_val, X_test).
    # OUTPUTS: pd.DataFrame arrays (X_train_clean, X_val_clean, X_test_clean).
    # --------------------------------------------------------------------------
    log_progress(4, TOTAL_STEPS, "Running Tier-1 Data Cleaner Pipeline", start_time)
    cleaner: CreditDataCleaner = CreditDataCleaner(
        numerical_features=NUM_COLS, categorical_features=CAT_COLS
    )

    # LEAKAGE SAFEGUARD: Fit parameters ONLY on training set, then transform others
    X_train_clean: pd.DataFrame = cleaner.fit_transform(X_train)
    X_val_clean: pd.DataFrame = cleaner.transform(X_val)
    X_test_clean: pd.DataFrame = cleaner.transform(X_test)

    # --------------------------------------------------------------------------
    # STEP 5: TIER-2 DYNAMIC WOE BINNING & ENCODING
    # --------------------------------------------------------------------------
    # PURPOSE: Convert raw variables into relative risk scales using Weight of Evidence (WOE).
    # INPUTS:  Imputed matrices (X_train_clean, X_val_clean, X_test_clean), pd.Series y_train.
    # OUTPUTS: pd.DataFrame arrays (X_train_woe, X_val_woe, X_test_woe) [Filtered to IV >= 0.02].
    # --------------------------------------------------------------------------
    log_progress(
        5, TOTAL_STEPS, "Running Tier-2 WOE Transformation & IV Filtering", start_time
    )
    bin_config: Dict[str, Any] = config["features"]["bin_config"]
    woe_transformer: WOETransformer = WOETransformer(
        numerical_features=NUM_COLS,
        categorical_features=CAT_COLS,
        bin_config=bin_config,
    )

    # LEAKAGE SAFEGUARD: Calculate mathematical boundaries ONLY using training data
    X_train_woe: pd.DataFrame = woe_transformer.fit_transform(X_train_clean, y_train)
    X_val_woe: pd.DataFrame = woe_transformer.transform(X_val_clean)
    X_test_woe: pd.DataFrame = woe_transformer.transform(X_test_clean)

    # --------------------------------------------------------------------------
    # STEP 6: MODEL HUẤN LUYỆN & TRAINING SET SANITY CHECK
    # --------------------------------------------------------------------------
    # PURPOSE: Optimize beta coefficients via Logistic Regression and audit basic model health.
    # INPUTS:  pd.DataFrame X_train_woe, pd.Series y_train, Dict config["model"].
    # OUTPUTS: Trained CreditModelTrainer engine.
    # --------------------------------------------------------------------------
    log_progress(
        6, TOTAL_STEPS, "Training Logistic Regression & Auditing Betas", start_time
    )
    model_trainer: CreditModelTrainer = CreditModelTrainer(model_config=config["model"])
    model_trainer.fit(X_train_woe, y_train)

    # SANITY CHECK: Predict back on training data to audit self-accuracy parameters
    y_train_pred: np.ndarray = model_trainer.predict_class(X_train_woe)
    y_train_prob: np.ndarray = model_trainer.predict_probability(X_train_woe)
    train_accuracy: float = np.mean(y_train_pred == y_train)
    print(f"\n[SANITY CHECK] Training Verification:")
    print(
        f"  -> Model Self-Accuracy on Training Population: {train_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------------------------
    # STEP 7: MLOPS ARTIFACT STORAGE PACKAGING & MULTI-POPULATION EXPORT
    # --------------------------------------------------------------------------
    # PURPOSE: Release evaluation packages, save processed datasets, and export the final model.
    # INPUTS:  All processed features, true targets, predicted classifications, and validation metrics.
    # OUTPUTS: Flat CSV arrays, structured JSON performance logs, PNG analysis curves, serialized PKL.
    # --------------------------------------------------------------------------
    log_progress(
        7, TOTAL_STEPS, "Generating Isolated Run Packages & Visualizations", start_time
    )

    current_run_dir: Path = get_run_directory(ARTIFACTS_DIR)
    print(f"[MLOPS] Storage Target Directory Activated: {current_run_dir.name}")

    # Construct distinct storage rooms
    (current_run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (current_run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (current_run_dir / "models").mkdir(parents=True, exist_ok=True)
    (current_run_dir / "data").mkdir(parents=True, exist_ok=True)
    (current_run_dir / "tables").mkdir(parents=True, exist_ok=True)

    # --- FINANCIAL SCORE SCALE CONVERSION ---
    print("[MLOPS] Activating Scorecard Scaling Transformation...")
    score_scaler = CreditScorecardScaler(scaling_config=config["scorecard_scaling"])
    score_scaler.fit(model_trainer=model_trainer, woe_transformer=woe_transformer)

    # Reconstruct string nominal matrices for lookup scoring via utilities layer
    X_train_bins = extract_structural_bins(X_train_clean, woe_transformer)
    X_val_bins = extract_structural_bins(X_val_clean, woe_transformer)
    X_test_bins = extract_structural_bins(X_test_clean, woe_transformer)

    # Execute absolute credit score computation
    train_scores: pd.Series = score_scaler.transform(X_train_bins)
    val_scores: pd.Series = score_scaler.transform(X_val_bins)
    test_scores: pd.Series = score_scaler.transform(X_test_bins)

    # --- PHYSICAL DATASETS EXPORTATION ---
    print("[MLOPS] Saving processed row-level datasets to storage records...")

    # 1. Global Workspace Sync (data/processed/) -> Pure imputation, no WOE, no score
    global_train_final = X_train_clean.copy()
    global_train_final[TARGET_COL] = y_train

    global_val_final = X_val_clean.copy()
    global_val_final[TARGET_COL] = y_val

    global_test_final = X_test_clean.copy()
    global_test_final[TARGET_COL] = y_test

    global_processed_dir: Path = Path("data/processed")
    global_processed_dir.mkdir(parents=True, exist_ok=True)

    global_train_final.to_csv(global_processed_dir / "train_clean.csv", index=False)
    global_val_final.to_csv(global_processed_dir / "val_clean.csv", index=False)
    global_test_final.to_csv(global_processed_dir / "test_clean.csv", index=False)
    print(
        f"  -> Universal cleaned data (No WOE) synchronized at: {global_processed_dir}/"
    )

    # 2. Historical Locked Tracking (artifacts/runs/.../data/) -> Retain WOE & Append Score
    train_historical = X_train_woe.copy()
    train_historical["credit_score"] = train_scores
    train_historical[TARGET_COL] = y_train

    val_historical = X_val_woe.copy()
    val_historical["credit_score"] = val_scores
    val_historical[TARGET_COL] = y_val

    test_historical = X_test_woe.copy()
    test_historical["credit_score"] = test_scores
    test_historical[TARGET_COL] = y_test

    train_historical.to_csv(
        current_run_dir / "data" / "train_woe_final.csv", index=False
    )
    val_historical.to_csv(current_run_dir / "data" / "val_woe_final.csv", index=False)
    test_historical.to_csv(current_run_dir / "data" / "test_woe_final.csv", index=False)
    print(
        f"  -> Historical run data (WOE & Credit Score) locked inside: {current_run_dir}/data/"
    )

    # Save training metrics rules & Scorecard Look-ups
    save_iv_scores(
        iv_scores=woe_transformer.iv_scores,
        file_path=current_run_dir / "metrics" / "baseline_iv_scores.json",
    )
    save_woe_tables(
        woe_dicts=woe_transformer.woe_dictionaries,
        folder_path=current_run_dir / "tables",
    )
    score_scaler.export_artifacts(folder_path=current_run_dir / "tables")
    score_scaler.plot_monotonic_barcharts(folder_path=current_run_dir / "plots")

    # --- POPULATION A: TRAINING PERFORMANCE REPORT ---
    train_report_path: Path = (
        current_run_dir / "metrics" / "train_evaluation_report.json"
    )
    save_classification_report(
        y_true=y_train,
        y_pred=y_train_pred,
        y_prob=y_train_prob,
        file_path=train_report_path,
    )

    # --- POPULATION B: VALIDATION PERFORMANCE PACKAGES ---
    print("[MLOPS] Visualizing Validation Performance Plots...")
    y_val_pred: np.ndarray = model_trainer.predict_class(X_val_woe)
    y_val_prob: np.ndarray = model_trainer.predict_probability(X_val_woe)

    val_report_path: Path = (
        current_run_dir / "metrics" / "validation_evaluation_report.json"
    )
    save_classification_report(
        y_true=y_val, y_pred=y_val_pred, y_prob=y_val_prob, file_path=val_report_path
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

    # --- POPULATION C: TESTING PERFORMANCE PACKAGES ---
    print("[MLOPS] Visualizing Testing Performance Plots...")
    y_test_pred: np.ndarray = model_trainer.predict_class(X_test_woe)
    y_test_prob: np.ndarray = model_trainer.predict_probability(X_test_woe)

    test_report_path: Path = current_run_dir / "metrics" / "test_evaluation_report.json"
    save_classification_report(
        y_true=y_test,
        y_pred=y_test_pred,
        y_prob=y_test_prob,
        file_path=test_report_path,
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
        "model_file_name", "baseline_logistic_model.pkl"
    )
    model_export_path: Path = current_run_dir / "models" / model_name
    joblib.dump(model_trainer, model_export_path)
    print(
        f"[MLOPS] Production model binary successfully locked at: {model_export_path}"
    )

    # ==============================================================================
    # FINAL EXECUTIVE PERFORMANCE AUDIT SUMMARIES
    # ==============================================================================
    print("\n======================================================================")
    print("📊 TRIPLE-POPULATION METRICS COMPLIANCE GRID")
    print("======================================================================")

    with open(train_report_path, "r") as trf, open(val_report_path, "r") as vf, open(
        test_report_path, "r"
    ) as tf:
        tr_scores: Dict[str, float] = json.load(trf)["scores"]
        v_scores: Dict[str, float] = json.load(vf)["scores"]
        t_scores: Dict[str, float] = json.load(tf)["scores"]

    print(f"  Metric Profile      | Train Dataset | Validation Set | Testing Dataset")
    print(f"  --------------------|---------------|----------------|----------------")
    print(
        f"  Kolmogorov-Smirnov  | {tr_scores['kolmogorov_smirnov_ks']:>13.4f} | {v_scores['kolmogorov_smirnov_ks']:>14.4f} | {t_scores['kolmogorov_smirnov_ks']:>15.4f}"
    )
    print(
        f"  F1-Classification   | {tr_scores['f1_score']:>13.4f} | {v_scores['f1_score']:>14.4f} | {t_scores['f1_score']:>15.4f}"
    )
    print(
        f"  Model Precision     | {tr_scores['precision']:>13.4f} | {v_scores['precision']:>14.4f} | {t_scores['precision']:>15.4f}"
    )
    print(
        f"  Recall/Sensitivity  | {tr_scores['recall_sensitivity']:>13.4f} | {v_scores['recall_sensitivity']:>14.4f} | {t_scores['recall_sensitivity']:>15.4f}"
    )
    print(
        f"  Overall Accuracy    | {tr_scores['accuracy']:>13.4f} | {v_scores['accuracy']:>14.4f} | {t_scores['accuracy']:>15.4f}"
    )

    print("\n======================================================================")
    print("📋 REGULATORY RISK COEFFICIENTS VALIDATION AUDIT")
    print("======================================================================")
    print(f"Base Intercept (Beta_0) : {model_trainer.intercept_:.4f}")

    all_betas_valid: bool = True
    for feature, beta in model_trainer.coefficients_.items():
        is_valid = beta < 0
        status_icon = "✅ VALID" if is_valid else "❌ CRITICAL ERROR"
        if not is_valid:
            all_betas_valid = False
        feature_iv: float = woe_transformer.iv_scores.get(feature, 0.0)
        print(
            f"  - {feature:<22} | Beta: {beta:.4f} ({status_icon}) | IV: {feature_iv:.4f}"
        )

    print("-" * 70)
    if all_betas_valid:
        print("🛡️ RISK STATUS: PASSED. Model complies with credit risk mathematics.")
    else:
        print("🚨 RISK STATUS: FAILED. Review positive features for trend reversal.")
    print("======================================================================")

    total_elapsed: float = time.time() - start_time
    print(
        f"\n[SUCCESS] End-to-End master pipeline executed cleanly in {total_elapsed:.2f} seconds!"
    )
    print(
        f"[MLOPS] All decoupled engineering packages are locked at: {current_run_dir}\n"
    )


if __name__ == "__main__":
    main()
