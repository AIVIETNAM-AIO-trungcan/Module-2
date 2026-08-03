"""
Inference Pipeline Engine - Credit Risk Scorecard Project
------------------------------------------------------
Serves as the core bridge between model training artifacts and deployment
interfaces. Refactored to fix URL downloading, type casting, numeric operations,
and guarantee safe fallback initialization to prevent application crashes.
"""

from io import StringIO
import os
from pathlib import Path
import zipfile
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download

from src.config import CONFIG_YAML_PATH, RUNS_DIR
from src.utils import extract_structural_bins

# Define Hugging Face Repository details
HF_REPO_ID: str = "trungcan94/AIO_moddule_2_model"


class CreditScorecardInferencePipeline:
    """Inference Engine class for scoring new credit applicant profiles."""

    def __init__(self, config_path: Path = CONFIG_YAML_PATH) -> None:
        """
        Initializes the inference pipeline. Features a fail-safe mechanism to prevent
        application crashes during Streamlit startup if model artifacts or configs are missing.
        """
        self.config_path: Path = config_path
        self.config: Dict[str, Any] = {}
        self.active_run_dir: Path = RUNS_DIR / "run_default"

        self.cleaner: Any = None
        self.woe_transformer: Any = None
        self.model: Any = None
        self.score_scaler: Any = None

        # Readiness flag to prevent crashes during the predict phase
        self.is_ready: bool = False

        # Fail-safe initialization block
        try:
            self.config = self._load_config()
            self.active_run_dir = self._resolve_active_run_dir()

            # Automatically download assets & model artifacts if missing
            self._ensure_assets_exist()
            self._ensure_artifacts_exist()
            self._load_artifacts()

            self.is_ready = True
            print("✅ [INFERENCE INITIALIZATION] Pipeline Engine is fully ready.")
        except Exception as e:
            print(
                f"⚠️ [SYSTEM FALLBACK WARNING] Pipeline initialized in standby mode due to error: {e}"
            )

    def _load_config(self) -> Dict[str, Any]:
        """Loads the YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"[CRITICAL] Configuration file not found at: {self.config_path}"
            )
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _resolve_active_run_dir(self) -> Path:
        """Determines which model run directory to use based on the configuration."""
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

        inf_config: Dict[str, Any] = self.config.get("inference", {})
        mode: str = inf_config.get("mode", "auto").lower()

        if mode == "auto":
            all_runs: List[Path] = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
            if not all_runs:
                default_run: Path = RUNS_DIR / "run_default"
                default_run.mkdir(parents=True, exist_ok=True)
                print(
                    f"⚠️ [INFERENCE WARNING] No runs found. Created default run directory: {default_run.name}"
                )
                return default_run

            latest_run: Path = max(all_runs, key=lambda d: d.stat().st_mtime)
            print(
                f"🤖 [INFERENCE INFO] Auto-detected latest model run: {latest_run.name}"
            )
            return latest_run

        elif mode == "manual":
            manual_run_name: Optional[str] = inf_config.get("manual_run_name")
            if not manual_run_name:
                raise ValueError(
                    "[CRITICAL] 'manual_run_name' must be specified when mode is set to 'manual'."
                )

            target_dir: Path = RUNS_DIR / manual_run_name
            target_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"🎯 [INFERENCE INFO] Manually specified model run: {target_dir.name}"
            )
            return target_dir

        else:
            raise ValueError(
                f"[CRITICAL] Invalid inference mode '{mode}'. Supported options are 'auto' or 'manual'."
            )

    def _ensure_assets_exist(self) -> None:
        """
        Checks if the asset directory exists and contains the required images.
        Downloads asset.zip from the Hugging Face Hub if missing.
        """
        project_root: Path = Path.cwd()
        asset_dir: Path = project_root / "asset"
        asset_dir.mkdir(parents=True, exist_ok=True)

        main_logo: Path = asset_dir / "capybara_main.png"

        if not main_logo.exists():
            print(
                "🌐 [CLOUD DOWNLOAD INFO] Asset images missing. Downloading asset.zip from HF Hub..."
            )
            try:
                downloaded_zip_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename="asset.zip",
                    repo_type="dataset",
                )
                print(
                    "📦 [CLOUD EXTRACTION] Extracting asset files directly into the asset directory..."
                )
                with zipfile.ZipFile(downloaded_zip_path, "r") as zip_ref:
                    zip_ref.extractall(asset_dir)

                files_found = os.listdir(asset_dir) if asset_dir.exists() else []
                print(
                    f"✅ [CLOUD SUCCESS] Mascot assets extracted successfully! Files available: {files_found}"
                )
            except Exception as e:
                print(
                    f"⚠️ [CLOUD DOWNLOAD ERROR] Failed to download or extract asset.zip: {e}"
                )

    def _download_artifacts_from_cloud(self, target_models_dir: Path) -> None:
        """
        Safely downloads the serialized model artifacts ZIP from the Hugging Face Hub.
        """
        target_models_dir.mkdir(parents=True, exist_ok=True)
        print(
            "🌐 [CLOUD DOWNLOAD INFO] Model artifacts missing. Downloading from Hugging Face Hub..."
        )

        try:
            downloaded_zip_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="model.zip",
                repo_type="dataset",
            )

            print("📦 [CLOUD EXTRACTION] Download completed. Extracting model files...")

            with zipfile.ZipFile(downloaded_zip_path, "r") as zip_ref:
                zip_ref.extractall(target_models_dir)

            print("✅ [CLOUD SUCCESS] Model artifacts successfully restored!")

        except Exception as e:
            raise RuntimeError(
                f"[CRITICAL DOWNLOAD ERROR] Failed to fetch model artifacts from Hugging Face. Error: {e}"
            )

    def _ensure_artifacts_exist(self) -> None:
        """Validates the presence of required model files, triggering a download if necessary."""
        models_dir: Path = self.active_run_dir / "models"
        model_name: str = self.config["model"]["logistic_regression"].get(
            "model_file_name", "champion_logistic_model.pkl"
        )

        required_artifacts: List[Path] = [
            models_dir / "cleaner.pkl",
            models_dir / "woe_transformer.pkl",
            models_dir / "score_scaler.pkl",
            models_dir / model_name,
        ]

        missing_count: int = sum(not p.exists() for p in required_artifacts)
        if missing_count > 0:
            print(
                f"⚠️ [INFERENCE WARNING] {missing_count} required artifact(s) missing in `{models_dir.name}`."
            )
            self._download_artifacts_from_cloud(models_dir)

    def _load_artifacts(self) -> None:
        """Loads the saved models, transformers, and scalers into memory."""
        models_dir: Path = self.active_run_dir / "models"

        cleaner_path: Path = models_dir / "cleaner.pkl"
        woe_path: Path = models_dir / "woe_transformer.pkl"
        scaler_path: Path = models_dir / "score_scaler.pkl"

        model_name: str = self.config["model"]["logistic_regression"].get(
            "model_file_name", "champion_logistic_model.pkl"
        )
        model_path: Path = models_dir / model_name

        if (
            not model_path.exists()
            and (models_dir / "baseline_logistic_model.pkl").exists()
        ):
            model_path = models_dir / "baseline_logistic_model.pkl"

        for artifact_path in [cleaner_path, woe_path, scaler_path, model_path]:
            if not artifact_path.exists():
                raise FileNotFoundError(
                    f"[CRITICAL MISSING FILE] Required pipeline artifact missing at: {artifact_path}"
                )

        self.cleaner = joblib.load(cleaner_path)
        self.woe_transformer = joblib.load(woe_path)
        self.score_scaler = joblib.load(scaler_path)
        self.model = joblib.load(model_path)
        print(
            "✅ [INFERENCE SUCCESS] All core artifacts loaded successfully into memory."
        )

    def _validate_and_preprocess_input(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Validates incoming data against required features and computes derived metrics."""
        df: pd.DataFrame = input_df.copy()
        num_cols: List[str] = self.config["features"]["numerical"]
        cat_cols: List[str] = self.config["features"]["categorical"]

        derived_cfg: Dict[str, Any] = self.config.get("derived_features", {})
        if "loan_percent_income_computed" in derived_cfg:
            cols: List[str] = derived_cfg["loan_percent_income_computed"]["columns"]
            precision: int = derived_cfg["loan_percent_income_computed"].get(
                "precision", 4
            )
            if cols[0] in df.columns and cols[1] in df.columns:
                num = df[cols[0]].to_numpy(dtype=float)
                denom = df[cols[1]].to_numpy(dtype=float)

                ratio = np.zeros_like(num)
                valid_mask = denom > 0
                ratio[valid_mask] = num[valid_mask] / denom[valid_mask]
                df["loan_percent_income_computed"] = ratio.round(precision)

        required_features: set = set(num_cols + cat_cols)
        existing_features: set = set(df.columns)
        missing_features: set = required_features - existing_features

        if missing_features:
            raise ValueError(
                f"[INPUT VALIDATION ERROR] Input data is missing required features: {missing_features}"
            )

        return df

    def _determine_decision(self, credit_score: float) -> Tuple[str, str]:
        """Maps a credit score to a final business decision and risk band."""
        thresholds: Dict[str, float] = self.config["scoring_thresholds"]
        app_score: float = thresholds["approval_score"]
        rev_score: float = thresholds["review_score"]

        if credit_score >= app_score:
            return "APPROVED", "Low Risk"
        elif credit_score >= rev_score:
            return "MANUAL_REVIEW", "Medium Risk"
        else:
            return "REJECTED", "High Risk"

    def _compute_score_breakdown(
        self, df_clean: pd.DataFrame, df_bins: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, int]]:
        """Calculates the point contribution of each feature towards the final score."""
        scaler: Any = self.score_scaler
        breakdowns: List[Dict[str, int]] = []
        df_target: pd.DataFrame = df_bins if df_bins is not None else df_clean

        points_map: Dict[str, Dict[str, int]] = getattr(scaler, "points_map", {})
        bin_edges: Dict[str, Any] = getattr(scaler, "bin_edges", {})

        sc_df: Optional[pd.DataFrame] = None
        if hasattr(scaler, "scorecard_table"):
            obj: Any = getattr(scaler, "scorecard_table")
            if isinstance(obj, pd.DataFrame):
                sc_df = obj
            elif isinstance(obj, str):
                try:
                    sc_df = pd.read_csv(StringIO(obj), sep=r"\s+", engine="python")
                except Exception:
                    sc_df = None

        for idx in range(len(df_target)):
            row_breakdown: Dict[str, int] = {}
            row_bins: pd.Series = df_target.iloc[idx]
            row_clean: pd.Series = df_clean.iloc[idx]

            for col in df_clean.columns:
                points: int = 0
                val_bin: str = str(row_bins.get(col, ""))
                val_clean: Any = row_clean.get(col, "")

                if col in points_map and isinstance(points_map[col], dict):
                    feature_map: Dict[str, int] = points_map[col]

                    if val_clean in feature_map:
                        points = feature_map[val_clean]
                    elif str(val_clean) in feature_map:
                        points = feature_map[str(val_clean)]
                    elif val_bin in feature_map:
                        points = feature_map[val_bin]
                    elif col in bin_edges:
                        edges = bin_edges[col]
                        try:
                            val_numeric = float(val_clean)
                            if isinstance(edges, (list, np.ndarray)) and not np.isnan(
                                val_numeric
                            ):
                                bin_idx: int = int(
                                    np.digitize([val_numeric], edges, right=False)[0]
                                    - 1
                                )
                                bin_idx = max(0, min(bin_idx, len(edges) - 2))
                                bin_key: str = f"Bin_{bin_idx}"
                                if bin_key in feature_map:
                                    points = feature_map[bin_key]
                        except (ValueError, TypeError):
                            pass

                if points == 0 and sc_df is not None and not sc_df.empty:
                    col_feature: str = (
                        "Feature" if "Feature" in sc_df.columns else "feature"
                    )
                    col_bin: Optional[str] = (
                        "Bin/Category"
                        if "Bin/Category" in sc_df.columns
                        else ("bin" if "bin" in sc_df.columns else None)
                    )
                    col_pts: Optional[str] = (
                        "Scaled_Points"
                        if "Scaled_Points" in sc_df.columns
                        else ("points" if "points" in sc_df.columns else None)
                    )

                    if col_feature and col_bin and col_pts:
                        matched: pd.DataFrame = sc_df[
                            (sc_df[col_feature] == col)
                            & (
                                (sc_df[col_bin].astype(str) == val_bin)
                                | (sc_df[col_bin].astype(str) == str(val_clean))
                            )
                        ]
                        if not matched.empty:
                            points = int(matched[col_pts].iloc[0])

                row_breakdown[col] = int(points)

            breakdowns.append(row_breakdown)

        return breakdowns

    def predict(self, input_df: pd.DataFrame) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Executes the scoring pipeline on new data.
        Will raise a controlled error if the pipeline is not correctly initialized.
        """
        if not self.is_ready:
            raise RuntimeError(
                "[INFERENCE BLOCKED] The model engine is not ready. This is likely due to missing artifacts, "
                "configuration files, or network failure during startup. Please check application logs."
            )

        df_validated: pd.DataFrame = self._validate_and_preprocess_input(input_df)

        df_clean: pd.DataFrame = self.cleaner.transform(df_validated)
        df_woe: pd.DataFrame = self.woe_transformer.transform(df_clean)
        df_bins: pd.DataFrame = extract_structural_bins(df_clean, self.woe_transformer)

        credit_scores: pd.Series = self.score_scaler.transform(df_bins)
        probabilities: np.ndarray = self.model.predict_probability(df_woe)

        is_single_record: bool = len(input_df) == 1

        if is_single_record:
            score_val: int = int(round(credit_scores.iloc[0]))
            prob_val: float = float(round(probabilities[0], 4))
            decision, risk_band = self._determine_decision(score_val)
            breakdown: Dict[str, int] = self._compute_score_breakdown(
                df_clean, df_bins
            )[0]

            return {
                "credit_score": score_val,
                "probability_of_default": prob_val,
                "decision": decision,
                "risk_band": risk_band,
                "score_breakdown": breakdown,
                "active_run_dir": str(self.active_run_dir.name),
            }
        else:
            result_df: pd.DataFrame = input_df.copy()
            result_df["credit_score"] = credit_scores.round().astype(int)
            result_df["probability_of_default"] = np.round(probabilities, 4)

            decisions_and_bands: List[Tuple[str, str]] = [
                self._determine_decision(s) for s in result_df["credit_score"]
            ]
            result_df["decision"] = [d[0] for d in decisions_and_bands]
            result_df["risk_band"] = [d[1] for d in decisions_and_bands]

            return result_df


if __name__ == "__main__":
    try:
        pipeline: CreditScorecardInferencePipeline = CreditScorecardInferencePipeline()
        if pipeline.is_ready:
            print("🚀 [SUCCESS] Inference Pipeline Engine initialized successfully!")
        else:
            print(
                "⚠️ [WARNING] Inference Pipeline initialized with errors. It will not process predictions."
            )
    except Exception as e:
        print(f"❌ [CRITICAL ERROR] Initialization Failed entirely: {e}")
