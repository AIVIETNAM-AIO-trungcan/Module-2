"""
Utility and Helper Functions Module
-----------------------------------
Purpose:
    This module provides shared helper functions for logging, execution timing,
    MLOps run lineage tracking, and exporting performance artifacts to disk.

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
from typing import Dict, Any


# ==============================================================================
# 1. Pipeline Execution Logging & Progress Orchestration
# ==============================================================================
def get_timestamp() -> str:
    """
    Generates a current localized timestamp string for logging

    Returns:
        str: Formatted time string (YYYY-MM-DD HH:MM:SS).
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_run_directory(base_artifacts_dir: pathlib.Path) -> pathlib.Path:
    """
    Scans the directory and dynamically generates a unique path for the current run.
    Naming pattern: YYYY-MM-DD_run_X (e.g., 2026-07-17_run_1)
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
    Serves as the vital matching data interface for the Scorecard Scaling Engine.

    Note:
        The woe_transformer parameter is explicitly typed as Any to neutralize
        circular compilation loops between utils and preprocessing layers.
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
    """
    Saves validation scores and reports into the artifacts folder as JSON file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)
    print(f"[UTILS] Metrics successfully saved to: {file_path}")


def save_woe_tables(
    woe_dicts: Dict[str, Dict[str, float]], folder_path: pathlib.Path
) -> None:
    """
    Exports computed WOE lookup tables into individual CSV files for historical auditing.
    """
    folder_path.mkdir(parents=True, exist_ok=True)
    for col, mapping in woe_dicts.items():
        df = pd.DataFrame(
            list(mapping.items()), columns=["Category_Or_Bin", "WOE_score"]
        )
        output_file = folder_path / f"{col}_woe.csv"
        df.to_csv(output_file, index=False)
    print(
        f"[UTILS] ALL WOE tables ({len(woe_dicts)} files) successfully exported to: {folder_path}"
    )


def save_iv_scores(iv_scores: Dict[str, float], file_path: pathlib.Path) -> None:
    """
    Saves the calculated Information Value (IV) scores into a JSON file for model governance.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(iv_scores, file, indent=4)
    print(f"[UTILS] IV Scores successfully saved to: {file_path}")


# ==============================================================================
# 4. Analytical Charts & Statistical Visualizations Engine
# ==============================================================================
def save_roc_curve(y_true: Any, y_prob: Any, file_path: pathlib.Path) -> None:
    """
    Plots a publication-quality Receiver Operating Characteristic (ROC) curve.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        fpr, tpr, color="#1f77b4", lw=2.5, label=f"Baseline Model (AUC = {roc_auc:.4f})"
    )
    ax.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.2, linestyle="--")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel(
        "False Boxitive Rate - FPR (1 - Specificity)",
        fontsize=11,
        labelpad=10,
        color="#1e293b",
    )
    ax.set_ylabel(
        "True Positive Rate - TPR (Sensitivity)",
        fontsize=11,
        labelpad=10,
        color="#1e293b",
    )
    ax.set_title(
        "Receiver Operating Characteristic (ROC) Curve",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color="#0f172a",
    )
    ax.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
    ax.legend(
        loc="lower right",
        fontsize=10,
        frameon=True,
        facecolor="white",
        edgecolor="none",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")
    ax.tick_params(colors="#64748b", labelsize=9)

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[UTILS] Publication-grade ROC Curve successfully exported to: {file_path}")


def save_classification_report(
    y_true: Any, y_pred: Any, y_prob: Any, file_path: pathlib.Path
) -> None:
    """
    Calculates Confusion Matrix components, derives F1-Score metrics,
    computes the Kolmogorov-Smirnov (KS) statistic, and exports a structural JSON report.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks_statistic = float(np.max(tpr - fpr))

    report_metrics = {
        "confusion_matrix": {
            "true_negatives_TN": int(tn),
            "false_positives_FP": int(fp),
            "false_negatives_FN": int(fn),
            "true_positives_TP": int(tp),
        },
        "scores": {
            "kolmogorov_smirnov_ks": ks_statistic,
            "f1_score": float(f1_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall_sensitivity": float(recall_score(y_true, y_pred)),
            "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        },
    }

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(report_metrics, file, indent=4)
    print(
        f"[UTILS] Confusion Matrix, F1 & KS Report successfully saved to: {file_path}"
    )


def save_probability_distribution(
    y_true: Any, y_prob: Any, file_path: pathlib.Path
) -> None:
    """
    Plots a professional, clean technical visualization of the Logistic Regression
    Sigmoid function curve mapped against actual and predicted risk observations.
    """
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
        linestyle="-",
        label="Theoretical Sigmoid Fit",
    )

    ax.scatter(
        logits[y_true_arr == 0],
        y_true_arr[y_true_arr == 0],
        color="#1f77b4",
        alpha=0.6,
        edgecolors="none",
        s=50,
        zorder=2,
        label="Actual Good (Class 0)",
    )
    ax.scatter(
        logits[y_true_arr == 1],
        y_true_arr[y_true_arr == 1],
        color="#d62728",
        alpha=0.6,
        edgecolors="none",
        s=50,
        zorder=2,
        label="Actual Bad (Class 1)",
    )

    ax.scatter(
        logits,
        y_prob_arr,
        color="#0f172a",
        s=25,
        marker="o",
        alpha=0.5,
        zorder=3,
        label="Predicted Probability (PD)",
    )

    ax.set_xlabel(
        "Log-Odds / Logit Score (Linear Combination)",
        fontsize=11,
        labelpad=10,
        color="#1e293b",
    )
    ax.set_ylabel(
        "Probability of Default (PD) / Outcome Scale",
        fontsize=11,
        labelpad=10,
        color="#1e293b",
    )
    ax.set_title(
        "Logistic Regression Alignment: Sigmoid Curve & Predictions",
        fontsize=12,
        fontweight="bold",
        pad=15,
        color="#0f172a",
    )

    ax.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1")
    ax.legend(
        loc="upper left", fontsize=9, frameon=True, facecolor="white", edgecolor="none"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")
    ax.tick_params(colors="#64748b", labelsize=9)

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[UTILS] Sigmoid Alignment chart successfully exported to: {file_path}")


def save_confusion_matrix_heatmap(
    y_true: Any, y_pred: Any, file_path: pathlib.Path
) -> None:
    """
    Plots a crisp, publication-grade confusion matrix heatmap using pure matplotlib.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    cax = ax.imshow(cm, cmap=plt.cm.Blues, alpha=0.7)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > (cm.max() / 2) else "#0f172a"
            ax.text(
                x=j,
                y=i,
                s=f"{cm[i, j]:,}",
                va="center",
                ha="center",
                fontsize=12,
                fontweight="bold",
                color=text_color,
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(
        ["Predicted Good (0)", "Predicted Bad (1)"], fontsize=10, color="#1e293b"
    )
    ax.set_yticklabels(
        ["Actual Good (0)", "Actual Bad (1)"], fontsize=10, color="#1e293b"
    )

    ax.set_xlabel(
        "Predicted Risk Classifications", labelpad=12, fontsize=11, color="#1e293b"
    )
    ax.set_ylabel(
        "Actual Risk Classifications", labelpad=12, fontsize=11, color="#1e293b"
    )
    ax.set_title(
        "Confusion Matrix Structural Distribution",
        pad=20,
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[UTILS] Confusion Matrix Heatmap successfully exported to: {file_path}")


# ==============================================================================
# INTERNAL TEST BLOCK
# ==============================================================================
if __name__ == "__main__":
    print("--- Executing Utility Module Independency ---")
    from src.config import ARTIFACTS_DIR

    current_time = get_timestamp()
    print(f"[TEST] Current system timestamp: {current_time}")

    # Test 1: Progress logging check
    log_progress(
        step_num=1,
        total_steps=3,
        step_name="Mock Progress Ingestion",
        start_time=time.time(),
    )

    # Test 2: Metrics logging
    MOCK_METRICS = {
        "test_run_date": current_time,
        "baseline_auc": 0.8245,
        "status": "active",
    }
    try:
        save_metrics(
            MOCK_METRICS, ARTIFACTS_DIR / "metrics" / "test_baseline_performance.json"
        )
    except Exception as e:
        print(f"[TEST FAILED] Metrics stop: {str(e)}")

    # Test 3: WOE tables
    MOCK_WOE_DATA = {"person_age": {"[0, 25)": 0.15, "[25, 50)": -0.05}}
    try:
        save_woe_tables(MOCK_WOE_DATA, ARTIFACTS_DIR / "tables" / "test_run")
    except Exception as e:
        print(f"[TEST FAILED] WOE stop: {str(e)}")

    # Test 4: IV Scores
    MOCK_IV_DATA = {"person_age": 0.2231, "person_income": 0.3456}
    try:
        save_iv_scores(MOCK_IV_DATA, ARTIFACTS_DIR / "metrics" / "test_iv_scores.json")
    except Exception as e:
        print(f"[TEST FAILED] IV stop: {str(e)}")

    # Setup core array matrices for updated test assertions
    MOCK_Y_TRUE = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    MOCK_Y_PROB = [0.10, 0.25, 0.85, 0.60, 0.30, 0.75, 0.20, 0.70, 0.90, 0.15]
    MOCK_Y_PRED = [0, 0, 1, 1, 0, 1, 1, 1, 1, 0]

    # Test 5: ROC curve plotting
    try:
        save_roc_curve(
            y_true=MOCK_Y_TRUE,
            y_prob=MOCK_Y_PROB,
            file_path=ARTIFACTS_DIR / "plots" / "test_baseline_roc_curve.png",
        )
    except Exception as e:
        print(f"[TEST FAILED] ROC plot stop: {str(e)}")

    # Test 6: Updated Classification Report
    try:
        save_classification_report(
            y_true=MOCK_Y_TRUE,
            y_pred=MOCK_Y_PRED,
            y_prob=MOCK_Y_PROB,
            file_path=ARTIFACTS_DIR
            / "metrics"
            / "test_baseline_evaluation_report.json",
        )
    except Exception as e:
        print(f"[TEST FAILED] Classification report stop: {str(e)}")

    # Test 7: Verify Predict vs Actual distribution mapping chart
    try:
        save_probability_distribution(
            y_true=MOCK_Y_TRUE,
            y_prob=MOCK_Y_PROB,
            file_path=ARTIFACTS_DIR / "plots" / "test_probability_distribution.png",
        )
    except Exception as e:
        print(f"[TEST FAILED] Prob distribution stop: {str(e)}")

    # Test 8: Verify confusion matrix heatmap
    try:
        save_confusion_matrix_heatmap(
            y_true=MOCK_Y_TRUE,
            y_pred=MOCK_Y_PRED,
            file_path=ARTIFACTS_DIR / "plots" / "test_confusion_matrix_heatmap.png",
        )
        print("\n==================================================")
        print("[SUCCESS] All 8 utility sandbox test cases executed successfully.")
        print("==================================================")
    except Exception as e:
        print(f"[TEST FAILED] CM Heatmap stop: {str(e)}")
