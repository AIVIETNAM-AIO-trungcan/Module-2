"""
Utility and Helper Functions Module
-----------------------------------
Purpose:
    This module provides shared helper functions for logging, execution timing,
    and exporting performance artifacts to disk.

Usages:
    Import helper functions directly into other scripts (e.g., 'from src.utils import get_timestamp').
"""

import json
import pathlib
from datetime import datetime
import pandas as pd
from typing import Dict, Any


# 1. Define time and logging helper functions
def get_timestamp() -> str:
    """
    Generates a current localized timestamp string for logging

    Returns:
        str: Formatted time string (YYYY-MM-DD HH:MM:SS).
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 2. Define artifact exporting helpers for model metrics
def save_metrics(metrics: Dict[str, Any], file_path: pathlib.Path) -> None:
    """
    Saves validation scores and reports into the artifacts folder as JSON file.
    Automatically creates the destination directory if it does not exist.

    Args:
        metrics (Dict[str, Any]): Dictionary containing score name and values
        file_path (pathlib.Path): Absolute destination path from src.config
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as file:
        json.dump(metrics, file, indent=4)
    print(f"[UTILS] Metrics successfully saved to: {file_path}")


# 3. Define artifact exporting helper for mathematical tables
def save_woe_tables(
    woe_dicts: Dict[str, Dict[str, float]], folder_path: pathlib.Path
) -> None:
    """
    Exports computed WOE lookup tables into individual CSV files for historical auditing.
    Automatically creates the destination directory if it does not exist.

    Args:
        woe_dicts: Dict[str, Dict[str,  float]]: The fitted woe dictionaries from the preprocessor
        folder_path: pathlib.Path: Absolute destination folder path from src.config.
    """
    folder_path.mkdir(parents=True, exist_ok=True)

    for col, mapping in woe_dicts.items():
        df = pd.DataFrame(
            list(mapping.items()), columns=["Category_Or_Bin", "WOE_score"]
        )
        output_file = folder_path / f"{col}_woe.csv"
        df.to_csv(output_file, index=False)

    print(
        f"[UTILS]  ALL WOE tables ({len(woe_dicts)} files) successfully exported to: {folder_path}"
    )


# INTERNAL TEST BLOCK
if __name__ == "__main__":
    print("--- Executing Utility Module Independency ---")

    # Test 1: Verify logging timestamp mechnism
    current_time = get_timestamp()
    print(f"[TEST] Current system timestamp: {current_time}")

    # Test 2: Verify path-safe metrics logginf with mock data
    from src.config import METRICS_DIR

    MOCK_METRICS = {
        "test_run_date": current_time,
        "baseline_auc": 0.8245,
        "status": "active",
    }
    try:
        save_metrics(MOCK_METRICS, METRICS_DIR / "test_baseline_performance.json")
        print("[TEST SUCCESS] Utility module passed all independent tests.")
    except Exception as e:
        print(f"[TEST FAILED] Stopped by: {type(e).__name__} | Details: {str(e)}")

    # Test 3: Verify mathematical WOE table exporting
    from src.config import ARTIFACTS_DIR

    MOCK_WOE_DATA = {
        "person_age": {"[0, 25)": 0.15, "[25, 50)": -0.05},
        "loan_intent": {"PERSONAL": 0.22, "EDUCATION": -0.12},
    }
    try:
        save_woe_tables(MOCK_WOE_DATA, ARTIFACTS_DIR / "tables" / "test_run")
        print("[TEST SUCCESS] Utility module passed all independent tests.")
    except Exception as e:
        print(f"[TEST FAILED] Stopped by: {type(e).__name__} | Details: {str(e)}")
