"""
Utility and Helper Functions Module
-----------------------------------
Purpose:
    This module provides shared helper functions for logging, execution timing,
    MLOps run lineage tracking, exporting performance artifacts to disk,
    computing financial calibration metrics, auditing scorecard monotonicity,
    and generating business strategy (Decisioning) artifacts.

Usages:
    Import helper functions directly into other scripts (e.g., 'from src.utils import get_timestamp').
"""

import json
import time
import pathlib
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

    # Pinpoint Optimal Cut-off on the curve if parameters are supplied
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
) -> None:
    """Plots Calibration Curve (Reliability Diagram)."""
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
            Mean_predicted_PD=("Predicted_PD", "mean"),
            Actual_bad_rate=("Actual", "mean"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        calib_table["Mean_predicted_PD"],
        calib_table["Actual_bad_rate"],
        marker="o",
        markersize=6,
        color="#1f77b4",
        lw=2,
        label="Observed",
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
    ax.set_title("Calibration Plot", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


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


def generate_approval_table(
    y_true: np.ndarray, y_prob: np.ndarray, scores: pd.Series
) -> pd.DataFrame:
    """
    Sweeps through score deciles to simulate business approval strategies.
    Computes Approval Rate, Approved Bad Rate, Bad Capture, and Good Rejection.
    """
    df = pd.DataFrame(
        {"Actual": np.asarray(y_true, dtype=int), "Score": scores.to_numpy()}
    )

    # Define cut-offs using dynamic percentiles to mimic real-world business sweeping
    cut_offs = np.percentile(df["Score"], [10, 30, 50, 70, 90])
    cut_offs = sorted(list(set(cut_offs)), reverse=True)

    results = []
    total_customers = len(df)
    total_bad = df["Actual"].sum()
    total_good = total_customers - total_bad

    for cutoff in cut_offs:
        approved_mask = df["Score"] >= cutoff
        rejected_mask = ~approved_mask

        approved_customers = approved_mask.sum()
        approved_bad = df.loc[approved_mask, "Actual"].sum()
        rejected_bad = df.loc[rejected_mask, "Actual"].sum()
        rejected_good = (~df.loc[rejected_mask, "Actual"].astype(bool)).sum()

        results.append(
            {
                "Score_Cutoff": cutoff,
                "Approval_Rate": (
                    approved_customers / total_customers if total_customers > 0 else 0
                ),
                "Approved_Bad_Rate": (
                    approved_bad / approved_customers if approved_customers > 0 else 0
                ),
                "Bad_Capture_Rate": rejected_bad / total_bad if total_bad > 0 else 0,
                "Good_Rejection_Rate": (
                    rejected_good / total_good if total_good > 0 else 0
                ),
            }
        )

    return pd.DataFrame(results)


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

    # Plot 1: Approval Rate
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        approval_table["Score_Cutoff"],
        approval_table["Approval_Rate"],
        marker="o",
        markersize=8,
        lw=2,
    )
    ax.set_title("Approval Rate by Credit Score Cut-off", fontsize=13, pad=15)
    ax.set_xlabel("Score cut-off")
    ax.set_ylabel("Approval rate")
    ax.grid(True, linestyle=":", alpha=0.6)
    for _, row in approval_table.iterrows():
        ax.annotate(
            f"{row['Approval_Rate']:.1%}",
            (row["Score_Cutoff"], row["Approval_Rate"]),
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
        approval_table["Score_Cutoff"],
        approval_table["Approved_Bad_Rate"],
        marker="o",
        markersize=8,
        lw=2,
    )
    ax.set_title("Approved Bad Rate by Credit Score Cut-off", fontsize=13, pad=15)
    ax.set_xlabel("Score cut-off")
    ax.set_ylabel("Bad rate of approved customers")
    ax.grid(True, linestyle=":", alpha=0.6)
    for _, row in approval_table.iterrows():
        ax.annotate(
            f"{row['Approved_Bad_Rate']:.1%}",
            (row["Score_Cutoff"], row["Approved_Bad_Rate"]),
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
        approval_table["Approval_Rate"],
        approval_table["Approved_Bad_Rate"],
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
    for _, row in approval_table.iterrows():
        ax.annotate(
            f"Score $\\geq$ {int(row['Score_Cutoff'])}",
            (row["Approval_Rate"], row["Approved_Bad_Rate"]),
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

    df = pd.DataFrame({"credit_score": test_scores, "target": y_test_true.to_numpy()})
    df["score_band"] = pd.qcut(df["credit_score"], q=10, duplicates="drop")
    score_band_audit = (
        df.groupby("score_band", observed=True)
        .agg(
            Number_of_observations=("target", "size"),
            Mean_Credit_Score=("credit_score", "mean"),
            Actual_Bad_Rate=("target", "mean"),
        )
        .reset_index()
    )

    print(f"  Score Band           | Count | Mean Score | Actual Bad Rate")
    print(f"  ---------------------|-------|------------|----------------")
    for _, row in score_band_audit.iterrows():
        print(
            f"  {str(row['score_band']):<20} | {row['Number_of_observations']:>5.0f} | {row['Mean_Credit_Score']:>10.1f} | {row['Actual_Bad_Rate']:>14.2%}"
        )
    print("======================================================================\n")
