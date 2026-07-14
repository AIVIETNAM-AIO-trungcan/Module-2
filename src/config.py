"""
Project Configuration Module
----------------------------
Purpose:
    This module acts as the central path manager for the entire Module 2: Credit Scoring project.
    This uses Python's 'pathlib' library to dunamically compute absolute paths base on relative  positions.

Usages:
    Import path variables directly into other scripts (e.g., 'from src.config import RAW_DATA_FILE').
"""

import pathlib

# 1. Locate the absolute path of this specific config.py file
CURRENT_FILE_PATH = pathlib.Path(__file__).resolve()

# 2. Define the project root directory
PROJECT_ROOT = CURRENT_FILE_PATH.parent.parent

# 3. Define paths to main subdirectories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREPROCESSORED_DIR = ARTIFACTS_DIR / "preprocessors"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

# 4. Define specific paths for data files and serialized models
RAW_DATA_FILE = RAW_DATA_DIR / "data_raw.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_FILE = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"

MODEL_FILE = MODELS_DIR / "logistic_regression_baseline.pkl"
PREPROCESSOR_FILE = PREPROCESSORED_DIR / "preprocessor.pkl"

# Safety Check: Automatically  create directories if they do not exist
for path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    PREPROCESSORED_DIR,
    METRICS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Test execution to verify paths when running this script directly
if __name__ == "__main__":
    print(f"Project Root Directory: {PROJECT_ROOT}")
    print(f"Raw Data File Path: {RAW_DATA_FILE}")
