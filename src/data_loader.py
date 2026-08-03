"""
Data Loader and Splitting Module
--------------------------------
Purpose:
    This module is dedicated to the data acquisition and splitting phase.
    It loads the local raw dataset and splits it strictly into Train and Test sets.
    Supports dual-mode preprocessing interception to ensure Notebook Sync Verification.
"""

import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_FILE


def load_raw_training_data(
    file_path: Optional[str] = None, mode: str = "mlops_optimized"
) -> pd.DataFrame:
    """
    Loads the local raw credit dataset for model training.

    Args:
        file_path (Optional[str]): Custom file path. If None, use the configured default path.
        mode (str): Branching toggle ('notebook_match' or 'mlops_optimized').

    Returns:
        pd.DataFrame: Raw training dataset.

    Raises:
        FileNotFoundError: If the dataset cannot be found.
    """
    actual_path = file_path if file_path is not None else RAW_DATA_FILE

    try:
        df = pd.read_csv(actual_path)
        print(
            f"[LOADER] Raw training data loaded successfully. Initial Shape: {df.shape}"
        )

        # ======================================================================
        # BRANCH 1: NOTEBOOK MATCH (PRE-SPLIT INTERCEPTION)
        # ======================================================================
        # In the legacy notebook, specific anomalies were manually dropped BEFORE
        # the Train/Test split. This block ensures Notebook Sync Verification.
        if mode == "notebook_match":
            print(
                "  -> [LOADER MODE: NOTEBOOK MATCH] Intercepting data to hard-drop legacy anomalies before split..."
            )
            # 1. Drop NaN in person_emp_length
            df = df.dropna(subset=["person_emp_length"])
            # 2. Drop person_age > 100
            df = df[df["person_age"] <= 100]
            # 3. Drop person_emp_length > 100
            df = df[df["person_emp_length"] <= 100]
            print(
                f"  -> [LOADER] Post-Notebook-Cleaning Shape: {df.shape} (Expected Target: 31679 rows)"
            )

        return df

    except FileNotFoundError as e:
        print(f"[ERROR] Could not find raw data at {actual_path}.")
        raise e


def validate_target(df: pd.DataFrame, target_column: str) -> None:
    """
    Validate the target column before splitting.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Target column name.

    Raises:
        ValueError: If the target is invalid.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist.")

    target = df[target_column]

    if target.isna().any():
        raise ValueError("Target column contains missing values.")

    unique_values = set(target.unique())

    if unique_values != {0, 1}:
        raise ValueError(
            f"Target must contain only {{0, 1}}. Found: {sorted(unique_values)}"
        )

    if target.nunique() != 2:
        raise ValueError("Target must contain both classes.")


def split_train_test(
    df: pd.DataFrame,
    target_column: str,
    split_params: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset firmly into Train and Test sets.

    Args:
        df (pd.DataFrame): Complete training dataset.
        target_column (str): Target column name.
        split_params (Dict[str, Any]): Splitting configuration (test_size, random_state, stratify).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and Test datasets.
    """
    validate_target(df, target_column)

    test_size = split_params.get("test_size", 0.20)
    random_state = split_params.get("random_state", 42)
    should_stratify = split_params.get("stratify", True)

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column] if should_stratify else None,
    )

    print(
        "\n[LOADER] Stratified Train/Test Splitting Executed (Notebook Sync Verification):"
    )

    for name, dataset in (
        ("Train", df_train),
        ("Test", df_test),
    ):
        counts = dataset[target_column].value_counts().to_dict()

        # Calculate percentage distribution
        ratios = dataset[target_column].value_counts(normalize=True).to_dict()
        ratios_pct = {k: f"{v * 100:.2f}%" for k, v in ratios.items()}

        print(
            f"    - {name:<6}: {dataset.shape[0]:>6} rows | "
            f"Class count: {counts} | "
            f"Distribution: {ratios_pct}"
        )

    return df_train, df_test


if __name__ == "__main__":
    print("--- Executing Data Loader Module Independently ---")

    from src.config import CONFIG

    TARGET_COLUMN = CONFIG["target"]
    SPLIT_PARAMS = CONFIG["data"]["split_params"]
    PREP_MODE = CONFIG["data"].get("preprocessing_mode", "mlops_optimized")

    try:
        raw_data = load_raw_training_data(mode=PREP_MODE)

        train_set, test_set = split_train_test(
            raw_data,
            target_column=TARGET_COLUMN,
            split_params=SPLIT_PARAMS,
        )

    except Exception as e:
        print(f"[TEST FAILED] {type(e).__name__}: {e}")
