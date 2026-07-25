"""
Data Loader and Splitting Module
--------------------------------
Purpose:
    This module is dedicated to the training and validation phase.
    It loads the local raw dataset and splits it into Train, Validation and Test sets
    applying stratified splitting to maintain the class distribution of imbalanced data.
"""

import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_FILE


def load_raw_training_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the local raw credit dataset for model training.

    Args:
        file_path (Optional[str]):
            Custom file path. If None, use the configured default path.

    Returns:
        pd.DataFrame:
            Raw training dataset.

    Raises:
        FileNotFoundError:
            If the dataset cannot be found.
    """
    actual_path = file_path if file_path is not None else RAW_DATA_FILE

    try:
        df = pd.read_csv(actual_path)
        print(f"[LOADER] Raw training data loaded successfully. Shape: {df.shape}")
        return df

    except FileNotFoundError as e:
        print(f"[ERROR] Could not find raw data at {actual_path}.")
        raise e


def validate_target(df: pd.DataFrame, target_column: str) -> None:
    """
    Validate the target column before splitting.

    Args:
        df (pd.DataFrame):
            Input dataframe.

        target_column (str):
            Target column name.

    Raises:
        ValueError:
            If the target is invalid.
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


def split_train_val_test(
    df: pd.DataFrame,
    target_column: str,
    split_params: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into Train, Validation and Test sets.

    Args:
        df (pd.DataFrame):
            Complete training dataset.

        target_column (str):
            Target column name.

        split_params (Dict[str, Any]):
            Splitting configuration.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Train, Validation and Test datasets.
    """
    validate_target(df, target_column)

    test_size = split_params.get("test_size", 0.20)
    val_size = split_params.get("val_size", 0.16)
    random_state = split_params.get("random_state", 42)
    should_stratify = split_params.get("stratify", True)

    train_size = 1 - test_size - val_size

    assert abs(train_size + val_size + test_size - 1.0) < 1e-9, (
        "Split ratios must sum to 1.0."
    )

    relative_val_size = val_size / (train_size + val_size)

    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column] if should_stratify else None,
    )

    df_train, df_val = train_test_split(
        df_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=df_train_val[target_column] if should_stratify else None,
    )

    print("[LOADER] Stratified splitting finished:")

    for name, dataset in (
        ("Train", df_train),
        ("Validation", df_val),
        ("Test", df_test),
    ):
        counts = dataset[target_column].value_counts().to_dict()
        ratios = dataset[target_column].value_counts(normalize=True).round(4).to_dict()

        print(
            f"    - {name}: {dataset.shape} | "
            f"Class count: {counts} | "
            f"Distribution: {ratios}"
        )

    return df_train, df_val, df_test


if __name__ == "__main__":
    print("--- Executing Data Loader Module Independently ---")

    TARGET_COLUMN = "loan_status"

    SPLIT_PARAMS = {
        "test_size": 0.20,
        "val_size": 0.16,
        "random_state": 42,
        "stratify": True,
    }

    try:
        raw_data = load_raw_training_data()

        train_set, val_set, test_set = split_train_val_test(
            raw_data,
            target_column=TARGET_COLUMN,
            split_params=SPLIT_PARAMS,
        )

    except Exception as e:
        print(f"[TEST FAILED] {type(e).__name__}: {e}")