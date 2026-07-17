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

# 4. Define specific paths for data files and serialized models
CONFIG_YAML_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
    pipeline_config = yaml.safe_load(f)

RAW_FILE_NAME = pipeline_config["data"]["raw_file_name"]

RAW_DATA_FILE = RAW_DATA_DIR / RAW_FILE_NAME


# Safety Check: Automatically  create directories if they do not exist
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
