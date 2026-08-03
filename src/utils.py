"""
Utility and Helper Functions Module
-----------------------------------
Purpose:
    This module provides shared helper functions for logging, execution timing,
    MLOps run lineage tracking, exporting performance artifacts to disk,
    computing financial calibration metrics, auditing scorecard monotonicity,
    generating business strategy (Decisioning) artifacts, and cloud syncing.

Usages:
    Import helper functions directly into other scripts (e.g., 'from src.utils import get_timestamp').
"""

import os
import json
import time
import zipfile
import pathlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from huggingface_hub import HfApi
from typing import Dict, Any, Tuple


# ==============================================================================
# 1. Pipeline Execution Logging & Progress Orchestration
# ==============================================================================
def get_timestamp() -> str:
    """
    Generates a current localized timestamp string for logging
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_run_directory(base_artifacts_dir: pathlib.Path) -> pathlib.Path:
    """
    Scans the directory and dynamically generates a unique path for the current run.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    runs_master_dir = base_artifacts_dir / "runs"
    runs_master_dir.mkdir(parents=True, exist_ok=True)

    run_number = 1
    while True:
        potential_run_dir = runs_master_dir / f"{today_str}_run_{run_number}"
        if not potential_run_dir.exists():
            return potential_run_dir
        run_number += 1


def log_progress(
    step_num: int, total_steps: int, step_name: str, start_time: float
) -> None:
    """
    Calculates and prints the real-time completion percentage and Estimated Time Remaining (ETA).
    """
    elapsed_time: float = time.time() - start_time
    percent: float = (step_num / total_steps) * 100

    if step_num > 0:
        estimated_total_time: float = (elapsed_time / step_num) * total_steps
        eta: float = estimated_total_time - elapsed_time
        eta_str: str = f"{eta:.2f}s"
    else:
        eta_str = "Calculating..."

    print(f"\n[PROGRESS] {percent:>5.1f}% | Step {step_num}/{total_steps}: {step_name}")
    print(f"           Elapsed: {elapsed_time:.2f}s | ETA: {eta_str}")
    print("-" * 70)


# ==============================================================================
# 2. Structural Data Transformation & Mapping Utilities
# ==============================================================================
def extract_structural_bins(
    X_clean: pd.DataFrame, woe_transformer: Any
) -> pd.DataFrame:
    """
    Converts cleaned raw float profiles into nominal string bin labels.
    """
    X_bins = X_clean.copy()
    for col in woe_transformer.numerical_features:
        edges = woe_transformer.bin_edges[col]
        group_names = [f"Bin_{i}" for i in range(len(edges) - 1)]
        X_bins[col] = pd.cut(
            X_bins[col], bins=edges, include_lowest=True, labels=group_names
        ).astype(str)
        X_bins[col] = X_bins[col].replace("nan", "Missing")
    return X_bins


# ==============================================================================
# 3. Artifact Exporting Mechanics (Tabular & Governance Logs)
# ==============================================================================
def save_metrics(metrics: Dict[str, Any], file_path: pathlib.Path) -> None:
    """Saves validation scores and reports into JSON format."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)
    print(f"[UTILS] Metrics successfully saved to: {file_path}")


def save_woe_tables(
    woe_dicts: Dict[str, Dict[str, float]], folder_path: pathlib.Path
) -> None:
    """Exports computed WOE lookup tables into individual CSV files."""
    folder_path.mkdir(parents=True, exist_ok=True)
    for col, mapping in woe_dicts.items():
        df = pd.DataFrame(
            list(mapping.items()), columns=["Category_Or_Bin", "WOE_score"]
        )
        output_file = folder_path / f"{col}_woe.csv"
        df.to_csv(output_file, index=False)
    print(f"[UTILS] ALL WOE tables successfully exported to: {folder_path}")


def save_iv_scores(iv_scores: Dict[str, float], file_path: pathlib.Path) -> None:
    """Saves IV scores into a JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(iv_scores, file, indent=4)


# ==============================================================================
# 4. Analytical Charts & Statistical Visualizations Engine
# ==============================================================================
def save_roc_curve(
    y_true: Any,
    y_prob: Any,
    file_path: pathlib.Path,
    optimal_pd: float = None,
    optimal_score: float = None,
    optimal_sensitivity: float = None,
    optimal_specificity: float = None,
    title_suffix: str = "",
) -> None:
    """
    Plots a publication-quality ROC curve. Highlights the Optimal Youden Cut-off
    if thresholds are provided (Step 17+ Decisioning).
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        fpr,
        tpr,
        color="#1f77b4",
        lw=2.5,
        label=f"ROC Curve (AUC = {roc_auc:.4f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="#ff7f0e",
        lw=1.5,
        linestyle="--",
        label="Random classifier",
    )

    if optimal_pd is not None:
        idx = np.argmin(np.abs(thresholds - optimal_pd))
        ax.plot(
            fpr[idx],
            tpr[idx],
            marker="o",
            markersize=9,
            color="#1f77b4",
            label=f"Optimal Threshold\nPD cut-off = {optimal_pd:.4f}\nScore cut-off = {optimal_score:.1f}",
        )
        ax.annotate(
            f"Sensitivity = {optimal_sensitivity:.2%}\nSpecificity = {optimal_specificity:.2%}",
            xy=(fpr[idx], tpr[idx]),
            xytext=(fpr[idx] + 0.05, tpr[idx] - 0.12),
            arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.2),
            fontsize=10,
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, labelpad=10)

    title = f"Receiver Operating Characteristic Curve {title_suffix}"
    ax.set_title(title.strip(), fontsize=13, fontweight="bold", pad=15)

    ax.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
    ax.legend(loc="lower right", fontsize=10, frameon=True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[UTILS] ROC Curve {title_suffix} successfully exported to: {file_path}")


def save_classification_report(
    y_true: Any, y_pred: Any, y_prob: Any, file_path: pathlib.Path
) -> None:
    """Calculates Confusion Matrix, F1, KS, Gini, and exports JSON report."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks_statistic = float(np.max(tpr - fpr))
    roc_auc = float(auc(fpr, tpr))
    gini_index = 2.0 * roc_auc - 1.0

    report_metrics = {
        "confusion_matrix": {
            "true_negatives_TN": int(tn),
            "false_positives_FP": int(fp),
            "false_negatives_FN": int(fn),
            "true_positives_TP": int(tp),
        },
        "scores": {
            "kolmogorov_smirnov_ks": ks_statistic,
            "gini_index": gini_index,
            "roc_auc": roc_auc,
            "f1_score": float(f1_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall_sensitivity": float(recall_score(y_true, y_pred)),
            "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        },
    }

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(report_metrics, file, indent=4)


def save_probability_distribution(
    y_true: Any, y_prob: Any, file_path: pathlib.Path
) -> None:
    """Plots Logistic Regression Sigmoid curve mapped against predicted PDs."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)

    eps = 1e-6
    clipped_prob = np.clip(y_prob_arr, eps, 1.0 - eps)
    logits = np.log(clipped_prob / (1.0 - clipped_prob))

    logit_range = np.linspace(np.min(logits) - 1.5, np.max(logits) + 1.5, 300)
    sigmoid_curve = 1.0 / (1.0 + np.exp(-logit_range))

    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.plot(
        logit_range,
        sigmoid_curve,
        color="#64748b",
        lw=2,
        label="Theoretical Sigmoid Fit",
    )
    ax.scatter(
        logits[y_true_arr == 0],
        y_true_arr[y_true_arr == 0],
        color="#1f77b4",
        alpha=0.6,
        s=50,
        label="Actual Good (Class 0)",
    )
    ax.scatter(
        logits[y_true_arr == 1],
        y_true_arr[y_true_arr == 1],
        color="#d62728",
        alpha=0.6,
        s=50,
        label="Actual Bad (Class 1)",
    )
    ax.scatter(
        logits, y_prob_arr, color="#0f172a", s=25, alpha=0.5, label="Predicted PD"
    )

    ax.set_xlabel("Log-Odds / Logit Score")
    ax.set_ylabel("Probability of Default (PD)")
    ax.set_title(
        "Logistic Regression Alignment: Sigmoid Curve", fontsize=12, fontweight="bold"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_heatmap(
    y_true: Any, y_pred: Any, file_path: pathlib.Path
) -> None:
    """Plots confusion matrix heatmap using pure matplotlib."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    cax = ax.imshow(cm, cmap=plt.cm.Blues, alpha=0.7)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > (cm.max() / 2) else "#0f172a"
            ax.text(
                j,
                i,
                s=f"{cm[i, j]:,}",
                va="center",
                ha="center",
                fontsize=12,
                fontweight="bold",
                color=text_color,
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Good (0)", "Predicted Bad (1)"])
    ax.set_yticklabels(["Actual Good (0)", "Actual Bad (1)"])
    ax.set_title("Confusion Matrix", pad=20, fontsize=12, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_calibration_plot(
    y_true: Any, y_prob: Any, file_path: pathlib.Path, n_bins: int = 10
) -> pd.DataFrame:
    """Plots Calibration Curve and prints Calibration Table to console."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "Actual": np.asarray(y_true, dtype=int),
            "Predicted_PD": np.asarray(y_prob, dtype=float),
        }
    )
    df["PD_band"] = pd.qcut(df["Predicted_PD"], q=n_bins, duplicates="drop")

    calib_table = (
        df.groupby("PD_band", observed=True)
        .agg(
            Number_of_observations=("Actual", "size"),
            Mean_predicted_PD=("Predicted_PD", "mean"),
            Actual_bad_rate=("Actual", "mean"),
        )
        .reset_index()
    )

    # --- NOTEBOOK SYNC VERIFICATION ---
    print("\n" + "=" * 70)
    print("📊 CALIBRATION TABLE (TEST DATASET)")
    print("=" * 70)
    formatted_table = calib_table.copy()
    formatted_table["Mean_predicted_PD"] = formatted_table["Mean_predicted_PD"].apply(
        lambda x: f"{x:.2%}"
    )
    formatted_table["Actual_bad_rate"] = formatted_table["Actual_bad_rate"].apply(
        lambda x: f"{x:.2%}"
    )
    print(formatted_table.to_string(index=False))
    print("=" * 70 + "\n")

    # Plot logic
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        calib_table["Mean_predicted_PD"],
        calib_table["Actual_bad_rate"],
        marker="o",
        markersize=6,
        color="#1f77b4",
        lw=2,
        label="Observed by PD band",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#7f7f7f",
        lw=1.5,
        label="Perfect calibration",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Mean predicted probability of default")
    ax.set_ylabel("Actual bad rate")
    ax.set_title("Calibration Plot - Test Set", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

    return calib_table


def save_candidate_evaluation_charts(
    metrics_log: Dict[str, Any], folder_path: pathlib.Path
) -> None:
    """Generates 3 critical evaluation bar charts for Candidate Models."""
    folder_path.mkdir(parents=True, exist_ok=True)
    models, aucs, briers, features, colors = [], [], [], [], []

    for model_key, results in metrics_log.items():
        if model_key in ["champion", "one_se_threshold"]:
            continue
        models.append(results["summary"]["Model"])
        aucs.append(results["summary"]["OOF AUC"])
        briers.append(results["summary"]["OOF Brier"])
        features.append(results["summary"]["Number of features"])
        colors.append(
            "#4caf50" if model_key == metrics_log.get("champion") else "#d62728"
        )

    # Chart 1: AUC
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(models, aucs, color=colors, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, aucs):
        ax.text(
            val - 0.02,
            bar.get_y() + bar.get_height() / 2,
            f" {val:.4f}",
            va="center",
            ha="right",
            color="white",
            fontweight="bold",
        )
    ax.set_xlabel("Out-of-fold AUC")
    ax.set_title("OOF AUC Comparison", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(folder_path / "candidate_models_auc_comparison.png", dpi=300)
    plt.close()

    # Chart 2: Brier
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(models, briers, color=colors, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, briers):
        ax.text(
            val - 0.005,
            bar.get_y() + bar.get_height() / 2,
            f" {val:.4f}",
            va="center",
            ha="right",
            color="white",
            fontweight="bold",
        )
    ax.set_xlabel("Out-of-fold Brier score")
    ax.set_title("OOF Brier Score Comparison", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(folder_path / "candidate_models_brier_comparison.png", dpi=300)
    plt.close()

    # Chart 3: Complexity
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(models, features, color=colors, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, features):
        ax.text(
            val - 0.2,
            bar.get_y() + bar.get_height() / 2,
            f" {int(val)}",
            va="center",
            ha="right",
            color="white",
            fontweight="bold",
        )
    ax.set_xlabel("Number of features")
    ax.set_title("Model Complexity", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(folder_path / "candidate_models_complexity_comparison.png", dpi=300)
    plt.close()


# ==============================================================================
# 5. Champion Model Auto-Selection Logic (1-SE Rule)
# ==============================================================================
def evaluate_and_select_champion(
    candidate_results: Dict[str, Dict[str, Any]], champ_cfg: Dict[str, Any]
) -> tuple:
    """Selects Champion Model based on 1-SE Rule and Business Eligibility."""
    print("\n" + "=" * 80)
    print("🏆 [CHAMPION AUTO-SELECTION] EVALUATING CANDIDATES (1-SE RULE)")
    print("=" * 80)

    prediction_context = champ_cfg.get("prediction_context", "pre_decision")
    manual_override = champ_cfg.get("manual_override", None)

    summaries = []
    for m_key, results in candidate_results.items():
        summary = results["summary"].copy()
        summary["Model key"] = m_key
        summaries.append(summary)

    table = pd.DataFrame(summaries)

    if prediction_context == "pre_decision":
        eligible_keys = {
            "model_intermediate_predecision_full_financial",
            "model_2_explainable_predecision",
        }
    else:
        eligible_keys = set(table["Model key"])

    table["Business eligible"] = table["Model key"].isin(eligible_keys)
    eligible_table = table.loc[table["Business eligible"]].copy()

    if eligible_table.empty:
        raise ValueError("No models are eligible for deployment.")

    if manual_override:
        champion_key = manual_override
        one_se_threshold = None
        print(f"  -> [WARNING] Manual Override Forcing Champion: {champion_key}")
    else:
        best_index = eligible_table["Mean CV AUC"].idxmax()
        best_mean_auc = float(eligible_table.loc[best_index, "Mean CV AUC"])
        best_se_auc = float(eligible_table.loc[best_index, "SE CV AUC"])

        one_se_threshold = best_mean_auc - best_se_auc

        table["Within eligible 1-SE"] = table["Business eligible"] & (
            table["Mean CV AUC"] >= one_se_threshold
        )
        one_se_candidates = table.loc[table["Within eligible 1-SE"]].sort_values(
            ["Number of features", "Mean CV AUC", "OOF Brier"],
            ascending=[True, False, True],
        )

        selected_row = one_se_candidates.iloc[0]
        champion_key = selected_row["Model key"]

        print(f"  -> Eligible 1-SE Threshold: {one_se_threshold:.5f}")
        print(f"  -> Selected by 1-SE Parsimony: {selected_row['Model']}")

    print(f"  🏆 FINAL CHAMPION: {champion_key.upper()}")
    candidate_results["champion"] = champion_key
    candidate_results["one_se_threshold"] = one_se_threshold

    return champion_key, candidate_results


# ==============================================================================
# 6. Financial Calibration Audit (PDO Error Measurement)
# ==============================================================================
def calculate_calibration_error(
    y_prob: np.ndarray, actual_scores: pd.Series, scaling_config: Dict[str, Any]
) -> Dict[str, float]:
    """Computes financial calibration error (Continuous vs Discrete Scores)."""
    base_score = scaling_config.get("base_score", 600)
    base_odds = scaling_config.get("base_odds", 50.0)
    pdo = scaling_config.get("pdo", 20)

    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    eps = 1e-10
    clipped_prob = np.clip(y_prob, eps, 1.0 - eps)
    log_odds = np.log((1.0 - clipped_prob) / clipped_prob)

    theoretical_scores = offset + factor * log_odds
    actual_scores_arr = actual_scores.to_numpy()
    absolute_errors = np.abs(theoretical_scores - actual_scores_arr)

    return {
        "mean_absolute_error_points": float(np.mean(absolute_errors)),
        "max_absolute_error_points": float(np.max(absolute_errors)),
        "calibration_variance_ratio": float(
            np.var(actual_scores_arr) / np.var(theoretical_scores)
        ),
    }


# ==============================================================================
# 7. Decision Strategy & Business Impact Engine (Steps 17 - 20)
# ==============================================================================
def calculate_optimal_cutoff(y_true: Any, y_prob: Any) -> Dict[str, float]:
    """
    Identifies the optimal PD threshold by maximizing the Youden Index (J).
    J = Sensitivity (TPR) + Specificity (1 - FPR) - 1.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)

    return {
        "optimal_pd_threshold": float(thresholds[optimal_idx]),
        "sensitivity": float(tpr[optimal_idx]),
        "specificity": float(1 - fpr[optimal_idx]),
        "youden_index": float(youden_j[optimal_idx]),
    }


def build_cutoff_table(
    y_true: Any, credit_score: Any, thresholds: Any = None, optimal_score: float = None
) -> pd.DataFrame:
    """
    Generates a comprehensive approval policy evaluation table across multiple score cut-offs.
    If optimal_score is provided, it guarantees this specific score is evaluated.

    Convention:
        Score >= cut-off: Approve
        Score < cut-off: Reject
    """
    y_true = np.asarray(y_true, dtype=int)
    credit_score = np.asarray(credit_score, dtype=float)

    if len(y_true) != len(credit_score):
        raise ValueError("y_true and credit_score must have the same length.")

    if thresholds is None:
        thresholds = np.arange(
            np.floor(credit_score.min()),
            np.ceil(credit_score.max()) + 1,
            1,
        ).astype(float)

    # Append the optimal_score to the thresholds array to ensure its calculation
    if optimal_score is not None:
        thresholds = np.append(thresholds, optimal_score)
        thresholds = np.unique(thresholds)
        thresholds = np.sort(thresholds)[::-1]  # Sort in descending order

    total_bad = np.sum(y_true == 1)
    total_good = np.sum(y_true == 0)

    rows = []

    for cutoff in thresholds:
        approved = credit_score >= cutoff
        rejected = ~approved

        approved_count = int(approved.sum())
        rejected_count = int(rejected.sum())

        approved_bad_count = int(y_true[approved].sum())
        rejected_bad_count = int(y_true[rejected].sum())
        rejected_good_count = int(np.sum(y_true[rejected] == 0))

        approval_rate = approved_count / len(y_true) if len(y_true) > 0 else 0

        approved_bad_rate = (
            approved_bad_count / approved_count if approved_count > 0 else np.nan
        )
        rejected_bad_rate = (
            rejected_bad_count / rejected_count if rejected_count > 0 else np.nan
        )
        bad_capture_rate = rejected_bad_count / total_bad if total_bad > 0 else np.nan
        good_rejection_rate = (
            rejected_good_count / total_good if total_good > 0 else np.nan
        )

        rows.append(
            {
                "Score cut-off": float(cutoff),
                "Approval count": approved_count,
                "Approval rate": approval_rate,
                "Approved bad count": approved_bad_count,
                "Approved bad rate": approved_bad_rate,
                "Rejected count": rejected_count,
                "Rejected bad rate": rejected_bad_rate,
                "Bad capture rate": bad_capture_rate,
                "Good rejection rate": good_rejection_rate,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("Score cut-off", ascending=False)
        .reset_index(drop=True)
    )


def display_cutoff_table(
    cutoff_table: pd.DataFrame, optimal_score: float = None
) -> None:
    """
    Prints a formatted subset of the cutoff table to the console (Notebook Sync Verification).
    Always includes the Optimal Score (if provided) and filters rows around target approval rates.
    """
    print("\n" + "=" * 120)
    print("📋 BUSINESS STRATEGY: CUT-OFF TABLE SIMULATION (NOTEBOOK SYNC VERIFICATION)")
    print("=" * 120)

    target_rates = [0.4, 0.5, 0.6, 0.7, 0.8]
    display_rows = []

    # 1. Prioritize and fetch the Optimal Score row first
    idx_opt = -1
    if optimal_score is not None:
        idx_opt = (cutoff_table["Score cut-off"] - optimal_score).abs().idxmin()
        display_rows.append(cutoff_table.loc[idx_opt])

    # 2. Extract evenly distributed business target approval rates
    for rate in target_rates:
        idx = (cutoff_table["Approval rate"] - rate).abs().idxmin()

        # Skip this row to prevent duplicates if it heavily overlaps with the Optimal Score
        if idx == idx_opt:
            continue

        display_rows.append(cutoff_table.loc[idx])

    display_df = pd.DataFrame(display_rows).reset_index(drop=True)
    formatted_df = display_df.copy()

    # Safely convert 'Score cut-off' to a list of formatted strings to bypass pandas LossySetitemError
    formatted_scores = []
    for i, val in enumerate(display_df["Score cut-off"]):
        if optimal_score is not None and i == 0:
            formatted_scores.append(f"{val:.1f} (Optimal)")
        else:
            formatted_scores.append(f"{val:.0f}")

    formatted_df["Score cut-off"] = formatted_scores

    # Safely format string percentages
    formatted_df["Approval rate"] = formatted_df["Approval rate"].apply(
        lambda x: f"{x:.2%}"
    )
    formatted_df["Approved bad rate"] = formatted_df["Approved bad rate"].apply(
        lambda x: "nan%" if pd.isna(x) else f"{x:.2%}"
    )
    formatted_df["Rejected bad rate"] = formatted_df["Rejected bad rate"].apply(
        lambda x: "nan%" if pd.isna(x) else f"{x:.2%}"
    )
    formatted_df["Bad capture rate"] = formatted_df["Bad capture rate"].apply(
        lambda x: "nan%" if pd.isna(x) else f"{x:.2%}"
    )
    formatted_df["Good rejection rate"] = formatted_df["Good rejection rate"].apply(
        lambda x: "nan%" if pd.isna(x) else f"{x:.2%}"
    )

    print(formatted_df.to_string(index=False))
    print("=" * 120 + "\n")


def save_business_strategy_plots(
    approval_table: pd.DataFrame, folder_path: pathlib.Path
) -> None:
    """
    Generates 3 critical business decision trade-off plots:
    1. Approval Rate by Cut-off.
    2. Approved Bad Rate by Cut-off.
    3. The overall Business Trade-off curve.
    """
    folder_path.mkdir(parents=True, exist_ok=True)

    # Subsample data points to prevent cluttered plotting
    target_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    plot_points = []
    for rate in target_rates:
        idx = (approval_table["Approval rate"] - rate).abs().idxmin()
        plot_points.append(approval_table.loc[idx])
    plot_df = pd.DataFrame(plot_points)

    # Plot 1: Approval Rate
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        plot_df["Score cut-off"],
        plot_df["Approval rate"],
        marker="o",
        markersize=8,
        lw=2,
    )
    ax.set_title("Approval Rate by Credit Score Cut-off", fontsize=13, pad=15)
    ax.set_xlabel("Score cut-off")
    ax.set_ylabel("Approval rate")
    ax.grid(True, linestyle=":", alpha=0.6)
    for _, row in plot_df.iterrows():
        ax.annotate(
            f"{row['Approval rate']:.1%}",
            (row["Score cut-off"], row["Approval rate"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(folder_path / "approval_rate_by_cutoff.png", dpi=300)
    plt.close()

    # Plot 2: Approved Bad Rate
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        plot_df["Score cut-off"],
        plot_df["Approved bad rate"],
        marker="o",
        markersize=8,
        lw=2,
    )
    ax.set_title("Approved Bad Rate by Credit Score Cut-off", fontsize=13, pad=15)
    ax.set_xlabel("Score cut-off")
    ax.set_ylabel("Bad rate of approved customers")
    ax.grid(True, linestyle=":", alpha=0.6)
    for _, row in plot_df.iterrows():
        ax.annotate(
            f"{row['Approved bad rate']:.1%}",
            (row["Score cut-off"], row["Approved bad rate"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(folder_path / "approved_bad_rate_by_cutoff.png", dpi=300)
    plt.close()

    # Plot 3: Trade-off
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        plot_df["Approval rate"],
        plot_df["Approved bad rate"],
        marker="o",
        markersize=8,
        lw=2,
    )
    ax.set_title(
        "Trade-off between Approval Rate and Approved Bad Rate", fontsize=13, pad=15
    )
    ax.set_xlabel("Approval rate")
    ax.set_ylabel("Approved bad rate")
    ax.grid(True, linestyle=":", alpha=0.6)
    for _, row in plot_df.iterrows():
        ax.annotate(
            f"Score $\\geq$ {int(row['Score cut-off'])}",
            (row["Approval rate"], row["Approved bad rate"]),
            textcoords="offset points",
            xytext=(10, 5),
            ha="left",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(folder_path / "tradeoff_approval_vs_bad_rate.png", dpi=300)
    plt.close()
    print(f"[UTILS] Business Strategy Plots successfully exported to: {folder_path}")


def audit_scorecard_results(
    test_scores: pd.Series, y_test_prob: np.ndarray, y_test_true: pd.Series
) -> None:
    """Audits scorecard monotonicity and constraints."""
    score_pd_corr = np.corrcoef(test_scores, y_test_prob)[0, 1]
    if score_pd_corr >= 0:
        raise ValueError(
            f"[CRITICAL FAILURE] Correlation(Score, PD) = {score_pd_corr:.4f}. Score must be inversely proportional to PD!"
        )
    print(f"  -> [PASS] Correlation(Score, PD) Check: {score_pd_corr:.4f} < 0")

    print("\n======================================================================")
    print("📈 SCORE BAND MONOTONICITY AUDIT (TEST DATASET)")
    print("======================================================================")

    df = pd.DataFrame(
        {
            "credit_score": test_scores,
            "predicted_pd": y_test_prob,
            "target": y_test_true.to_numpy(),
        }
    )

    # Bin based on Predicted PD to match Notebook Calibration Table Logic
    df["pd_band"] = pd.qcut(df["predicted_pd"], q=10, duplicates="drop")

    score_band_audit = (
        df.groupby("pd_band", observed=True)
        .agg(
            Number_of_observations=("target", "size"),
            Mean_Predicted_PD=("predicted_pd", "mean"),
            Actual_Bad_Rate=("target", "mean"),
            Mean_Credit_Score=("credit_score", "mean"),
        )
        .reset_index()
    )

    print(f"  PD Band              | Count | Mean Score | Actual Bad Rate")
    print(f"  ---------------------|-------|------------|----------------")
    for _, row in score_band_audit.iterrows():
        print(
            f"  {str(row['pd_band']):<20} | {row['Number_of_observations']:>5.0f} | {row['Mean_Credit_Score']:>10.1f} | {row['Actual_Bad_Rate']:>14.2%}"
        )
    print("======================================================================\n")


# ==============================================================================
# 8. Scorecard Table Generation (Notebook Sync)
# ==============================================================================
def display_scorecard_table(
    woe_dicts: Dict[str, Dict[str, float]],
    model_coefficients: Dict[str, float],
    intercept: float,
    factor: float,
    offset: float,
) -> pd.DataFrame:
    """
    Generates and prints the detailed scorecard table containing WOE, Coefficients,
    Raw Points, and Rounded Points for each bin, aligning with Notebook outputs.

    Args:
        woe_dicts: Dictionary containing mapping of bins to WOE values.
        model_coefficients: Dictionary containing logistic regression coefficients.
        intercept: The intercept of the fitted model.
        factor: Scorecard scaling factor (PDO / ln(2)).
        offset: Scorecard scaling offset.

    Returns:
        pd.DataFrame: Formatted scorecard dataframe.
    """
    base_points = offset - (factor * intercept)

    print("\n" + "=" * 80)
    print("📋 DETAILED SCORECARD TABLE BY BIN (NOTEBOOK SYNC VERIFICATION)")
    print("=" * 80)
    print(f"Base Points: {base_points:.4f} (Rounded: {int(round(base_points))})\n")

    rows = []
    for feature, bin_mapping in woe_dicts.items():
        if feature not in model_coefficients:
            continue
        coef = model_coefficients[feature]
        for bin_label, woe_val in bin_mapping.items():
            raw_points = -factor * coef * woe_val
            rounded_points = int(round(raw_points))
            rows.append(
                {
                    "Variable": feature,
                    "Bin": bin_label,
                    "WOE": woe_val,
                    "Coefficient": coef,
                    "Raw points": raw_points,
                    "Rounded points": rounded_points,
                }
            )

    scorecard_df = pd.DataFrame(rows)

    # Format and print the table to console
    formatted_df = scorecard_df.copy()
    formatted_df["WOE"] = formatted_df["WOE"].apply(lambda x: f"{x:.6f}")
    formatted_df["Coefficient"] = formatted_df["Coefficient"].apply(
        lambda x: f"{x:.6f}"
    )
    formatted_df["Raw points"] = formatted_df["Raw points"].apply(lambda x: f"{x:.4f}")

    print(formatted_df.to_string(index=False))
    print("=" * 80 + "\n")

    return scorecard_df


# ==============================================================================
# 9. Deployment & Cloud Sync Integration
# ==============================================================================
def package_and_upload_artifacts(run_dir: pathlib.Path, config: Dict[str, Any]) -> None:
    """
    Packages production model binaries into model.zip and syncs with Hugging Face Hub.

    Args:
        run_dir (pathlib.Path): The directory containing the current run's artifacts.
        config (Dict[str, Any]): Configuration dictionary containing deployment settings.
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


# ==============================================================================
# 10. Notebook-Sync Audit Logs & Mathematical Precision Checks
# ==============================================================================
def display_coefficient_stability(
    candidate_results: Dict[str, Any], champion_key: str
) -> None:
    """
    Displays the cross-validation coefficient stability table for the champion model
    and warns if any feature has a negative coefficient rate below 80%,
    syncing behavior with notebook logic.

    Args:
        candidate_results (Dict[str, Any]): Dictionary containing CV results for all models.
        champion_key (str): The key of the selected champion model.
    """
    final_cv_stability = candidate_results[champion_key]["coefficient_stability"]

    print("\n" + "=" * 70)
    print(f"📊 COEFFICIENT STABILITY AUDIT (NOTEBOOK SYNC VERIFICATION)")
    print("=" * 70)

    formatted_stability = final_cv_stability.copy()
    formatted_stability["Mean_coefficient"] = formatted_stability[
        "Mean_coefficient"
    ].apply(lambda x: f"{x:.6f}")
    formatted_stability["SD_coefficient"] = formatted_stability["SD_coefficient"].apply(
        lambda x: f"{x:.6f}"
    )
    formatted_stability["Negative_fold_rate"] = formatted_stability[
        "Negative_fold_rate"
    ].apply(lambda x: f"{x:.1%}")

    print(formatted_stability.to_string(index=False))

    sign_warning = final_cv_stability.loc[
        final_cv_stability["Negative_fold_rate"] < 0.80
    ]
    if sign_warning.empty:
        print("\n  -> [PASS] No variables with negative coefficient rate below 80%.")
    else:
        print("\n  -> ⚠️ [WARNING] Variables requiring coefficient sign review:")
        print(sign_warning.to_string(index=False))
    print("=" * 70 + "\n")


def display_champion_model_summary(champion_model: Any) -> None:
    """
    Prints the GLM statistical summary and detailed coefficient metrics
    (Standard Error, Z-statistic, P-value, Confidence Intervals, Odds Ratio)
    for the champion model to the console.

    Args:
        champion_model (Any): The fitted CreditModelTrainer object.
    """
    print("\n" + "=" * 80)
    print("📊 CHAMPION MODEL LOGISTIC REGRESSION SUMMARY (NOTEBOOK SYNC VERIFICATION)")
    print("=" * 80)

    # Print default statsmodels summary if available
    if hasattr(champion_model, "model_") and hasattr(champion_model.model_, "summary"):
        print(champion_model.model_.summary())

    print("\n  -- DETAILED COEFFICIENT STATISTICS & ODDS RATIOS --")

    # Print detailed metrics extracted from model audit
    coef_stats = pd.DataFrame(
        champion_model.audit_report_.get("coefficient_statistics", [])
    )
    if not coef_stats.empty:
        formatted_coefs = coef_stats.copy()
        for col in [
            "Coefficient",
            "Standard error",
            "P-value",
            "CI lower",
            "CI upper",
            "Odds ratio",
        ]:
            if col in formatted_coefs.columns:
                formatted_coefs[col] = formatted_coefs[col].apply(lambda x: f"{x:.6f}")
        if "Z statistic" in formatted_coefs.columns:
            formatted_coefs["Z statistic"] = formatted_coefs["Z statistic"].apply(
                lambda x: f"{x:.4f}"
            )

        print(formatted_coefs.to_string(index=False))
    print("=" * 80 + "\n")


def verify_score_precision(
    X_test_constant: pd.DataFrame,
    model_params: pd.Series,
    test_probability: np.ndarray,
    offset: float,
    factor: float,
) -> None:
    """
    Calculates the maximum difference between credit scores derived directly
    from logits versus scores derived from probabilities to ensure mathematical precision.

    Args:
        X_test_constant (pd.DataFrame): Test features with constant added.
        model_params (pd.Series): The parameters (coefficients) of the fitted model.
        test_probability (np.ndarray): The predicted probabilities.
        offset (float): The scorecard offset value.
        factor (float): The scorecard scaling factor.
    """
    test_linear_predictor = np.asarray(X_test_constant @ model_params, dtype=float)
    test_score_from_logit = offset - factor * test_linear_predictor

    # Calculate score from probability
    eps = 1e-8
    clipped_prob = np.clip(np.asarray(test_probability, dtype=float), eps, 1 - eps)
    logit_pd = np.log(clipped_prob / (1 - clipped_prob))
    test_score_from_probability = offset - factor * logit_pd

    max_diff = float(
        np.max(np.abs(test_score_from_logit - test_score_from_probability))
    )

    print("\n" + "-" * 70)
    print(f"🔢 SCORE CALCULATION PRECISION CHECK (NOTEBOOK SYNC VERIFICATION)")
    print(
        f"  -> Maximum difference between logit score and probability score: {max_diff}"
    )
    if max_diff > 1e-8:
        warnings.warn("Score calculation methods do not match (error > 1e-8).")
    else:
        print("  -> [PASS] Mathematical precision verified.")
    print("-" * 70 + "\n")
