"""
Main Execution Pipeline - Credit Risk Scorecard Project
------------------------------------------------------
Orchestrates data loading, stratified isolation, two-tier preprocessing
safeguards, dynamic WOE transformation, tri-branch feature selection,
champion model auto-selection (1-SE Rule via 10-Fold CV), multi-population
artifact packages, and cloud sync.
"""

from dotenv import load_dotenv

load_dotenv()  # Auto-loads environment variables from .env file

import os
import time
import yaml
import joblib
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from typing import Dict, List, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, log_loss

# Import localized modules from the src factory
from src.data_loader import load_raw_training_data, split_train_test
from src.preprocessing import CreditDataCleaner, WOETransformer
from src.feature_selection import CreditFeatureSelector
from src.model import CreditModelTrainer
from src.scorecard import CreditScorecardScaler
from src.config import (
    ARTIFACTS_DIR,
    CONFIG_YAML_PATH,
    RAW_DATA_FILE,
    PROCESSED_DATA_DIR,
)
from src.utils import (
    get_run_directory,
    save_roc_curve,
    save_classification_report,
    save_iv_scores,
    save_woe_tables,
    save_probability_distribution,
    save_confusion_matrix_heatmap,
    save_calibration_plot,
    save_candidate_evaluation_charts,
    log_progress,
    evaluate_and_select_champion,
    calculate_calibration_error,
    audit_scorecard_results,
    calculate_optimal_cutoff,
    build_cutoff_table,
    display_cutoff_table,
    save_business_strategy_plots,
    package_and_upload_artifacts,
    display_coefficient_stability,
    display_champion_model_summary,
    verify_score_precision,
    display_scorecard_table,
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
    config_path: Path = CONFIG_YAML_PATH
    if not config_path.exists():
        raise FileNotFoundError(
            f"[CRITICAL] Configuration file missing at: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    NUM_COLS: List[str] = config["features"]["numerical"]
    CAT_COLS: List[str] = config["features"]["categorical"]
    TARGET_COL: str = config["target"]

    # EXTRACT PREPROCESSING MODE (Crucial for Branching Logic)
    PREP_MODE: str = config["data"].get("preprocessing_mode", "mlops_optimized")

    # --------------------------------------------------------------------------
    # STEP 2: RAW DATA ACQUISITION
    # --------------------------------------------------------------------------
    log_progress(
        2, TOTAL_STEPS, f"Acquiring Raw Data Registry (Mode: {PREP_MODE})", start_time
    )
    data_path: Path = RAW_DATA_FILE

    # Passing the mode allows the data loader to drop legacy anomalies BEFORE splitting
    df_raw: pd.DataFrame = load_raw_training_data(file_path=data_path, mode=PREP_MODE)

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

    # Strictly Train and Test (No Validation Set)
    df_train, df_test = split_train_test(
        df=df_raw, target_column=TARGET_COL, split_params=split_params
    )

    X_train: pd.DataFrame = df_train.drop(columns=[TARGET_COL]).copy()
    y_train: pd.Series = df_train[TARGET_COL].copy()

    X_test: pd.DataFrame = df_test.drop(columns=[TARGET_COL]).copy()
    y_test: pd.Series = df_test[TARGET_COL].copy()

    # --------------------------------------------------------------------------
    # STEP 4: TIER-1 CLEANING PIPELINE (STATISTICAL & BUSINESS CONSTRAINTS)
    # --------------------------------------------------------------------------
    log_progress(
        4,
        TOTAL_STEPS,
        f"Running Tier-1 Data Cleaner Pipeline (Mode: {PREP_MODE})",
        start_time,
    )

    cleaner: CreditDataCleaner = CreditDataCleaner(
        numerical_features=NUM_COLS,
        categorical_features=CAT_COLS,
        cleaning_config=config.get("cleaning_config", {}),
        validation_config=config.get("validation", {}),
        mode=PREP_MODE,
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

    # --- 4.2 TESTING SET AUDIT ---
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
    X_test_woe: pd.DataFrame = woe_transformer.transform(X_test_clean)

    # --------------------------------------------------------------------------
    # STEP 6: TIER-3 TRI-BRANCH FEATURE SELECTION
    # --------------------------------------------------------------------------
    log_progress(
        6, TOTAL_STEPS, "Running Tier-3 Tri-Branch Feature Selection", start_time
    )

    selection_config = config["features"].get("selection_config", {})
    feature_selector = CreditFeatureSelector(selection_config=selection_config)

    # Returns a dict of DataFrames mapped to 3 Candidate Structures
    X_train_sel_dict: Dict[str, pd.DataFrame] = feature_selector.fit_transform(
        X_train_woe, y_train, woe_transformer.iv_scores
    )
    X_test_sel_dict: Dict[str, pd.DataFrame] = feature_selector.transform(X_test_woe)

    # --------------------------------------------------------------------------
    # STEP 7: 10-FOLD CV MODEL EVALUATION & CHAMPION AUTO-SELECTION
    # --------------------------------------------------------------------------
    log_progress(
        7, TOTAL_STEPS, "Evaluating Models via CV & Selecting Champion", start_time
    )

    cv_folds = split_params.get("cv_folds", 10)
    random_state = split_params.get("random_state", 42)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Pre-calculate identical splits to ensure fair comparison across all candidates
    cv_splits = list(cv.split(X_train_clean, y_train))
    candidate_results = {}

    for model_key, X_train_subset in X_train_sel_dict.items():
        model_label = model_key.replace("_", " ").title()
        trainer = CreditModelTrainer(model_config=config["model"])

        # Execute rigorous 10-Fold evaluation and statistical audits
        results = trainer.evaluate_cv(
            X=X_train_subset,
            y=y_train,
            cv_splits=cv_splits,
            model_key=model_key,
            model_label=model_label,
        )
        candidate_results[model_key] = results

    # Apply 1-SE rule to select Champion based on OOF AUC and Parsimony
    champ_cfg = selection_config.get("champion_selection", {})
    champion_key, metrics_log = evaluate_and_select_champion(
        candidate_results=candidate_results,
        champ_cfg=champ_cfg,
    )

    # --- NOTEBOOK SYNC VERIFICATION: Check Coefficient Stability for Champion ---
    display_coefficient_stability(metrics_log, champion_key)

    # Fit the Champion Model formally on the ENTIRE training subset to extract final coefficients
    champion_model = CreditModelTrainer(model_config=config["model"])
    champion_model.fit(X_train_sel_dict[champion_key], y_train)

    # --- NOTEBOOK SYNC VERIFICATION: Display Model Summary & Odds Ratios ---
    display_champion_model_summary(champion_model)

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

    # --- VISUALIZATION: MODEL COMPARISON BAR CHARTS ---
    save_candidate_evaluation_charts(
        metrics_log=metrics_log,
        folder_path=current_run_dir / "plots",
    )

    # --- FINANCIAL SCORE SCALE CONVERSION ---
    print(f"\n[MLOPS] Activating Scorecard Scaling for Champion ({champion_key})...")
    scaling_cfg = config["scorecard_scaling"]
    print(
        f"  -> Base Score: {scaling_cfg['base_score']} at Odds {scaling_cfg['base_odds']}:1"
    )
    print(f"  -> PDO: {scaling_cfg['pdo']}")

    score_scaler = CreditScorecardScaler(scaling_config=scaling_cfg)
    score_scaler.fit(model_trainer=champion_model, woe_transformer=woe_transformer)

    print(f"  -> Calculated Factor: {score_scaler.factor:.6f}")
    print(f"  -> Calculated Offset: {score_scaler.offset:.6f}")

    # --- NOTEBOOK SYNC VERIFICATION: Display Scorecard Bin Table ---
    # Safely extract coefficients from the audit report
    coef_stats = champion_model.audit_report_.get("coefficient_statistics", [])
    model_coefficients = {stat["Variable"]: stat["Coefficient"] for stat in coef_stats}

    # Safely extract intercept (Statsmodels typically uses 'const')
    intercept_value = float(model_coefficients.get("const", 0.0))

    display_scorecard_table(
        woe_dicts=woe_transformer.woe_dictionaries,
        model_coefficients=model_coefficients,
        intercept=intercept_value,
        factor=score_scaler.factor,
        offset=score_scaler.offset,
    )

    # Extract nominal bin strings for historical records matching the scaling logic
    X_train_bins = woe_transformer.transform_to_bins(X_train_clean)
    X_test_bins = woe_transformer.transform_to_bins(X_test_clean)

    train_scores: pd.Series = score_scaler.transform(X_train_bins)
    test_scores: pd.Series = score_scaler.transform(X_test_bins)

    # Generate exact probabilities from the FINAL fitted champion model
    X_train_champ = X_train_sel_dict[champion_key]
    X_test_champ = X_test_sel_dict[champion_key]
    y_train_prob_champ = champion_model.predict_probability(X_train_champ)
    y_test_prob_champ = champion_model.predict_probability(X_test_champ)
    y_oof_prob_champ = metrics_log[champion_key]["oof_probability"]

    # --- NOTEBOOK SYNC VERIFICATION: Score Precision Audit ---
    X_test_champ_const = sm.add_constant(X_test_champ, has_constant="add")
    verify_score_precision(
        X_test_constant=X_test_champ_const,
        model_params=pd.Series(model_coefficients),
        test_probability=y_test_prob_champ,
        offset=score_scaler.offset,
        factor=score_scaler.factor,
    )

    # --- BUSINESS DECISION STRATEGY & OPTIMAL CUT-OFF (STEPS 17-20) ---
    print(
        "\n[MLOPS] Activating Business Decision Strategy & Optimal Cut-off Analysis..."
    )

    # 1. Calculate Youden Index from OOF Train probabilities to prevent leakage
    optimal_cutoff_metrics = calculate_optimal_cutoff(
        y_true=y_train, y_prob=y_oof_prob_champ
    )
    optimal_pd = optimal_cutoff_metrics["optimal_pd_threshold"]

    # 2. Convert Optimal PD to Optimal Credit Score Cut-off
    factor = scaling_cfg["pdo"] / np.log(2)
    offset = scaling_cfg["base_score"] - factor * np.log(scaling_cfg["base_odds"])
    optimal_score = offset - factor * np.log(optimal_pd / (1.0 - optimal_pd))

    print(f"  -> Optimal PD Cut-off   : {optimal_pd:.4f}")
    print(f"  -> Optimal Score Cut-off: {optimal_score:.1f}")

    # 3. Generate Approval Strategy Table & Plots on Test Data
    print("  -> Generating Detailed Cut-off Strategy Table...")

    cutoff_table = build_cutoff_table(
        y_true=y_test,
        credit_score=test_scores,
        optimal_score=optimal_score,
    )

    # --- NOTEBOOK SYNC VERIFICATION: Display Cut-off Table ---
    display_cutoff_table(cutoff_table=cutoff_table, optimal_score=optimal_score)

    save_business_strategy_plots(
        approval_table=cutoff_table, folder_path=current_run_dir / "plots"
    )

    # Save the FULL table to CSV for business analysts
    cutoff_table.to_csv(
        current_run_dir / "tables" / "detailed_cutoff_strategy.csv", index=False
    )

    # --- FINANCIAL CALIBRATION AUDIT ---
    # Compare final model's probabilities against final model's exact scorecard points
    calibration_metrics = calculate_calibration_error(
        y_prob=y_train_prob_champ,
        actual_scores=train_scores,
        scaling_config=config["scorecard_scaling"],
    )

    with open(
        current_run_dir / "metrics" / "pdo_calibration_error.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(calibration_metrics, f, indent=4)

    # --- PHYSICAL DATASETS EXPORTATION & AUDIT (STEPS 17 & 18) ---
    print("\n[MLOPS] Saving processed row-level datasets and computing odds...")

    # Calculate Good:Bad Odds: Odds = (1 - PD) / PD
    train_odds = (1.0 - y_train_prob_champ) / np.clip(y_train_prob_champ, 1e-10, 1.0)
    test_odds = (1.0 - y_test_prob_champ) / np.clip(y_test_prob_champ, 1e-10, 1.0)

    train_historical = X_train_sel_dict[champion_key].copy()
    train_historical["predicted_pd"] = y_train_prob_champ
    train_historical["good_bad_odds"] = train_odds
    train_historical["credit_score"] = train_scores
    train_historical[TARGET_COL] = y_train

    test_historical = X_test_sel_dict[champion_key].copy()
    test_historical["predicted_pd"] = y_test_prob_champ
    test_historical["good_bad_odds"] = test_odds
    test_historical["credit_score"] = test_scores
    test_historical[TARGET_COL] = y_test

    # Execute Scorecard Audit (Correlation & Score Bands)
    audit_scorecard_results(
        test_scores=test_scores, y_test_prob=y_test_prob_champ, y_test_true=y_test
    )

    # Save to disk (Scored Historical)
    train_historical.to_csv(
        current_run_dir / "data" / f"train_scored_{champion_key}.csv", index=False
    )
    test_historical.to_csv(
        current_run_dir / "data" / f"test_scored_{champion_key}.csv", index=False
    )

    print("[MLOPS] Exporting Raw Train/Test Splits for Engineering Audits...")

    # 1. Saving in MLOps Run for tracking version
    df_train.to_csv(current_run_dir / "data" / "train_raw_split.csv", index=False)
    df_test.to_csv(current_run_dir / "data" / "test_raw_split.csv", index=False)

    # 2. Saving in 'data/processed'
    processed_dir: Path = PROCESSED_DATA_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(processed_dir / "train_raw_split.csv", index=False)
    df_test.to_csv(processed_dir / "test_raw_split.csv", index=False)

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

    # Export Cross-Validation evaluation grid for all candidate models
    with open(
        current_run_dir / "metrics" / "cross_validation_results_grid.json",
        "w",
        encoding="utf-8",
    ) as f:
        # Convert DataFrames to dicts for JSON serialization
        json_ready_log = {}
        for k, v in metrics_log.items():
            if k in ["champion", "one_se_threshold"]:
                json_ready_log[k] = v
            else:
                json_ready_log[k] = {
                    "summary": v["summary"],
                    "coefficient_stability": v["coefficient_stability"].to_dict(
                        orient="records"
                    ),
                    "fold_metrics": v["fold_metrics"].to_dict(orient="records"),
                }
        json.dump(json_ready_log, f, indent=4)

    with open(
        current_run_dir / "metrics" / "feature_selection_audit.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(feature_selector.audit_report_, f, indent=4)

    # Export Champion Model Mathematical Audit (Coefficients, P-values, Confidence Intervals)
    with open(
        current_run_dir / "metrics" / "champion_model_math_audit.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(champion_model.audit_report_, f, indent=4)

    save_iv_scores(
        woe_transformer.iv_scores,
        current_run_dir / "metrics" / "baseline_iv_scores.json",
    )
    save_woe_tables(woe_transformer.woe_dictionaries, current_run_dir / "tables")

    score_scaler.export_artifacts(folder_path=current_run_dir / "tables")
    score_scaler.plot_monotonic_barcharts(folder_path=current_run_dir / "plots")

    # --- POPULATION EVALUATIONS ---
    # IMPORTANT: Predict classes using the OPTIMAL PD Cut-off instead of default 0.5
    y_train_pred_optimal = (y_oof_prob_champ >= optimal_pd).astype(int)
    y_test_pred_optimal = (y_test_prob_champ >= optimal_pd).astype(int)

    save_classification_report(
        y_true=y_train,
        y_pred=y_train_pred_optimal,
        y_prob=y_oof_prob_champ,
        file_path=current_run_dir / "metrics" / "train_evaluation_report.json",
    )

    save_classification_report(
        y_true=y_test,
        y_pred=y_test_pred_optimal,
        y_prob=y_test_prob_champ,
        file_path=current_run_dir / "metrics" / "test_evaluation_report.json",
    )

    # Generate ROC Curve with the identified Optimal Score Cut-off
    save_roc_curve(
        y_true=y_test,
        y_prob=y_test_prob_champ,
        file_path=current_run_dir / "plots" / "test_roc_curve.png",
        optimal_pd=optimal_pd,
        optimal_score=optimal_score,
        optimal_sensitivity=optimal_cutoff_metrics["sensitivity"],
        optimal_specificity=optimal_cutoff_metrics["specificity"],
        title_suffix="with Locked OOF Cut-off",
    )

    save_probability_distribution(
        y_true=y_test,
        y_prob=y_test_prob_champ,
        file_path=current_run_dir / "plots" / "test_probability_distribution.png",
    )
    save_confusion_matrix_heatmap(
        y_true=y_test,
        y_pred=y_test_pred_optimal,
        file_path=current_run_dir / "plots" / "test_confusion_matrix_heatmap.png",
    )

    # --- NOTEBOOK SYNC VERIFICATION: Call Calibration Plot which also prints the table ---
    save_calibration_plot(
        y_true=y_test,
        y_prob=y_test_prob_champ,
        file_path=current_run_dir / "plots" / "test_calibration_curve.png",
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
    print(f"📊 METRICS COMPLIANCE GRID (CHAMPION: {champion_key.upper()})")
    print("======================================================================")
    with open(
        current_run_dir / "metrics" / "train_evaluation_report.json", "r"
    ) as trf, open(
        current_run_dir / "metrics" / "test_evaluation_report.json", "r"
    ) as tf:
        tr_scores = json.load(trf)["scores"]
        t_scores = json.load(tf)["scores"]

    train_brier = metrics_log[champion_key]["summary"]["OOF Brier"]
    test_brier = brier_score_loss(y_test, y_test_prob_champ)
    train_logloss = metrics_log[champion_key]["summary"]["OOF Log loss"]
    test_logloss = log_loss(y_test, y_test_prob_champ)

    print(f"  Metric Profile      | Train (OOF)   | Testing Dataset")
    print(f"  --------------------|---------------|----------------")
    print(
        f"  ROC-AUC             | {tr_scores['roc_auc']:>13.4f} | {t_scores['roc_auc']:>15.4f}"
    )
    print(
        f"  Gini Index          | {tr_scores['gini_index']:>13.4f} | {t_scores['gini_index']:>15.4f}"
    )
    print(
        f"  Kolmogorov-Smirnov  | {tr_scores['kolmogorov_smirnov_ks']:>13.4f} | {t_scores['kolmogorov_smirnov_ks']:>15.4f}"
    )
    print(f"  Brier Score         | {train_brier:>13.4f} | {test_brier:>15.4f}")
    print(f"  Log Loss            | {train_logloss:>13.4f} | {test_logloss:>15.4f}")
    print(f"  --------------------|---------------|----------------")
    print(
        f"  Precision           | {tr_scores['precision']:>13.4f} | {t_scores['precision']:>15.4f}"
    )
    print(
        f"  Recall (Sensitivity)| {tr_scores['recall_sensitivity']:>13.4f} | {t_scores['recall_sensitivity']:>15.4f}"
    )
    print(
        f"  F1-Classification   | {tr_scores['f1_score']:>13.4f} | {t_scores['f1_score']:>15.4f}"
    )
    print(
        f"  Overall Accuracy    | {tr_scores['accuracy']:>13.4f} | {t_scores['accuracy']:>15.4f}"
    )

    print("\n======================================================================")
    print("📋 REGULATORY RISK COEFFICIENTS VALIDATION AUDIT")
    print("======================================================================")
    print(f"Base Intercept (Beta_0) : {champion_model.intercept_:.4f}")

    coef_stats = champion_model.audit_report_.get("coefficient_statistics", [])
    all_passed = True
    for stat in coef_stats:
        if stat["Variable"] == "const":
            continue
        status = stat["Expected WOE sign"]
        icon = "✅ PASS" if status == "Pass" else "⚠️ REVIEW"
        if status != "Pass":
            all_passed = False
        print(
            f"  - {stat['Variable']:<25} | Beta: {stat['Coefficient']:.4f} | P-value: {stat['P-value']:.4f} ({icon})"
        )

    print("-" * 70)
    if all_passed:
        print(
            "🛡️ RISK STATUS: PASSED. All features conform to expected economic logic (Negative Betas)."
        )
    else:
        print(
            "🚨 RISK STATUS: WARNING. Some features have positive betas and require underwriter review."
        )
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
