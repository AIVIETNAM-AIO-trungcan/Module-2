"""
Project Configuration Module
----------------------------
Purpose:
    This module centralizes project paths and loads configuration values
    from the project's config.yaml file. It exposes commonly used paths
    and validation parameters for use throughout the pipeline.

Usages:
    Import configuration variables directly into other modules
    (e.g., 'from src.config import RAW_DATA_FILE, AGE_MIN').
"""

import pathlib
import yaml

# 1. Locate the absolute path of this specific config.py file
CURRENT_FILE_PATH = pathlib.Path(__file__).resolve()

# 2. Define the project root directory
PROJECT_ROOT = CURRENT_FILE_PATH.parent.parent

# 3. Define paths to main subdirectories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"

# 4. Define specific paths for data files and serialized models
CONFIG_YAML_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

RAW_FILE_NAME = CONFIG["data"]["raw_file_name"]

RAW_DATA_FILE = RAW_DATA_DIR / RAW_FILE_NAME

# 5. Numerical Validation Rules

AGE_MIN = CONFIG["validation"]["person_age"]["min"]
AGE_MAX = CONFIG["validation"]["person_age"]["max"]

INCOME_MIN = CONFIG["validation"]["person_income"]["min"]

EMP_LENGTH_MIN = CONFIG["validation"]["person_emp_length"]["min"]
EMP_LENGTH_MAX = CONFIG["validation"]["person_emp_length"]["max"]

LOAN_AMOUNT_MIN = CONFIG["validation"]["loan_amnt"]["min"]

INTEREST_RATE_MIN = CONFIG["validation"]["loan_int_rate"]["min"]

CREDIT_HISTORY_MIN = CONFIG["validation"]["cb_person_cred_hist_length"]["min"]


# Safety Check: Automatically create directories if they do not exist
for path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Test execution to verify paths when running this script directly
if __name__ == "__main__":
    print(f"Project Root Directory: {PROJECT_ROOT}")
    print(f"Dynamic Raw Data Path:  {RAW_DATA_FILE}")
