"""
Data Loader and Splitting Module
--------------------------------
Purpose:
    This module is dedicated to the training and validation phase.
    It loads the local raw dataset and splits it into Train, Validation and Test sets
    Applying stratified splitting to maintain the class distribution of imbalanced data
"""

import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_FILE


def load_raw_training_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the local raw credit dataset for model training.

    Returns:
        pd.DataFrame: The complete raw dataset for training.

    Raises:
        FileNotFoundError: If the raw data file does not exist at the configured path.
    """
    try:
        actual_path = file_path if file_path is not None else RAW_DATA_FILE
        df = pd.read_csv(actual_path)
        print(f"[LOADER] Raw training data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError as e:
        print(
            f"[ERROR] Could not find raw data at {actual_path}. Please check data/raw/folder."
        )
        raise e


def split_train_val_test(
    df: pd.DataFrame, target_column: str, split_params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the training dataset into Train-train, Validation, and Test sets.
    Uses stratified splitting based on the target column to handle imbalanced data.

    Args:
        df (pd.DataFrame): The input complete training dataset
        target_column (str): The name of the label/target column (in this project: 'loan_status')
        train_size (float): Proportion for Train-train (Default: 0.64/ 64%)
        val_size (float): Proportion for Validation (Default: 0.16/ 16%)
        test_size (float): Proportion for Test (Default: 0.2/ 20%)
        random_state (int): Random seed for reproducibility (Default: 42)

    Return:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (df_train, df_val, df_test)
    """
    test_size = split_params.get("test_size", 0.2)
    val_size = split_params.get("val_size", 0.16)
    random_state = split_params.get("random_state", 42)
    should_stratify = split_params.get("stratify", True)
    train_size = 1 - test_size - val_size

    # Quick sanity check on the ratios
    assert (
        abs((train_size + test_size + val_size) - 1.0) < 1e-9
    ), "Ratios must sum up to 1.0"

    # Calculate the remaining ratio for validation after isolating the test set
    relative_val_size = val_size / (train_size + val_size)

    # 1. First Split: Isolate the Test set using stratify on the target column
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column] if should_stratify else None,
    )  # Maintains class distribution in the Test set

    # 2. Second Split: Separate Train-train and Validation using stratify on target
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=df_train_val[target_column] if should_stratify else None,
    )  # Maintain class distribution in the Train/Val sets

    # Log information to verify the target distribution across all splits
    print(f"[LOADER] Stratified splitting finished:")
    for name, dataset in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        ratio = dataset[target_column].value_counts(normalize=True).to_dict()
        print(f"    -{name} Set: {dataset.shape} | Target distribution: {ratio}")

    return df_train, df_val, df_test


if __name__ == "__main__":
    print("--- Executing Data Loader Module Independently ---")
    try:
        raw_data = load_raw_training_data()

        # ASSUMPTION: Replace 'target' in this project: 'loan_status'
        TARGET_COL = "loan_status"

        MOCK_PARAMS = {
            "test_size": 0.2,
            "val_size": 0.16,
            "random_state": 42,
            "stratify": True,
        }

        train_set, val_set, test_set = split_train_val_test(
            raw_data, target_column=TARGET_COL, split_params=MOCK_PARAMS
        )
    except Exception as e:
        print(f"[TEST FAILED] Stopped by: {type(e).__name__}")
