"""
Inference Pipeline Engine - Credit Risk Scorecard Project
------------------------------------------------------
Serves as the core bridge between model training artifacts and deployment
interfaces (Streamlit UI, Batch CSV processing, or REST APIs). Handles dynamic
artifact loading (auto/manual), input validation, multi-tier feature transformations,
credit score scaling, score breakdown (waterfall attribution), and automated decisioning.
"""

from io import StringIO
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import yaml

from src.config import CONFIG_YAML_PATH, RUNS_DIR
from src.utils import extract_structural_bins


class CreditScorecardInferencePipeline:
    """Inference Engine class for scoring new credit applicant profiles."""

    def __init__(self, config_path: Path = CONFIG_YAML_PATH) -> None:
        """Initializes the inference engine by loading central configuration settings

        and active model artifacts into memory.

        Args:
            config_path (Path): File path pointing to central config.yaml.
        """
        self.config_path: Path = config_path
        self.config: Dict[str, Any] = self._load_config()
        self.active_run_dir: Path = self._resolve_active_run_dir()

        # Serialized artifact placeholders
        self.cleaner: Any = None
        self.woe_transformer: Any = None
        self.model: Any = None
        self.score_scaler: Any = None

        # Load models into memory
        self._load_artifacts()

    def _load_config(self) -> Dict[str, Any]:
        """Loads central project configuration settings from YAML.

        Returns:
            Dict[str, Any]: Dictionary representation of configuration settings.

        Raises:
            FileNotFoundError: If the YAML config file path does not exist.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"[CRITICAL] Configuration file not found at: {self.config_path}"
            )
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _resolve_active_run_dir(self) -> Path:
        """Resolves target artifact directory based on 'inference' settings in config.yaml.

        Returns:
            Path: Absolute or relative Path pointing to the target run directory.

        Raises:
            FileNotFoundError: If the runs directory or target run folder is missing.
            ValueError: If an invalid inference mode is specified.
        """
        if not RUNS_DIR.exists():
            raise FileNotFoundError(
                f"[CRITICAL] Artifact runs directory missing at: {RUNS_DIR}"
            )

        inf_config: Dict[str, Any] = self.config.get("inference", {})
        mode: str = inf_config.get("mode", "auto").lower()

        if mode == "auto":
            all_runs: List[Path] = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
            if not all_runs:
                raise FileNotFoundError(
                    f"[CRITICAL] No run directories found inside: {RUNS_DIR}"
                )

            latest_run: Path = max(all_runs, key=lambda d: d.stat().st_mtime)
            print(f"🤖 [INFERENCE] Auto-detected latest model run: {latest_run.name}")
            return latest_run

        elif mode == "manual":
            manual_run_name: Optional[str] = inf_config.get("manual_run_name")
            if not manual_run_name:
                raise ValueError(
                    "[CRITICAL] 'manual_run_name' must be specified when mode is set to 'manual'."
                )

            target_dir: Path = RUNS_DIR / manual_run_name
            if not target_dir.exists():
                raise FileNotFoundError(
                    f"[CRITICAL] Specified manual run directory not found: {target_dir}"
                )

            print(f"🎯 [INFERENCE] Manually loaded model run: {target_dir.name}")
            return target_dir

        else:
            raise ValueError(
                f"[CRITICAL] Invalid inference mode '{mode}'. Supported options are 'auto' or 'manual'."
            )

    def _load_artifacts(self) -> None:
        """Loads serialized PKL pipeline artifacts from the active run directory into memory.

        Raises:
            FileNotFoundError: If any required serialized artifact file is missing.
        """
        models_dir: Path = self.active_run_dir / "models"

        cleaner_path: Path = models_dir / "cleaner.pkl"
        woe_path: Path = models_dir / "woe_transformer.pkl"
        scaler_path: Path = models_dir / "score_scaler.pkl"

        model_name: str = self.config["model"]["logistic_regression"].get(
            "model_file_name", "baseline_logistic_model.pkl"
        )
        model_path: Path = models_dir / model_name

        for artifact_path in [cleaner_path, woe_path, scaler_path, model_path]:
            if not artifact_path.exists():
                raise FileNotFoundError(
                    f"[CRITICAL] Required pipeline artifact missing at: {artifact_path}"
                )

        self.cleaner = joblib.load(cleaner_path)
        self.woe_transformer = joblib.load(woe_path)
        self.score_scaler = joblib.load(scaler_path)
        self.model = joblib.load(model_path)
        print("✅ [INFERENCE] All core artifacts loaded successfully into memory.")

    def _validate_and_preprocess_input(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Validates feature presence and computes dynamic derived features if missing.

        Args:
            input_df (pd.DataFrame): Raw feature inputs provided by client or application.

        Returns:
            pd.DataFrame: Validated DataFrame populated with derived feature calculations.

        Raises:
            ValueError: If required baseline input features are missing.
        """
        df: pd.DataFrame = input_df.copy()
        num_cols: List[str] = self.config["features"]["numerical"]
        cat_cols: List[str] = self.config["features"]["categorical"]

        # Compute derived features dynamically based on config rules
        derived_cfg: Dict[str, Any] = self.config.get("derived_features", {})
        if "loan_percent_income_computed" in derived_cfg:
            cols: List[str] = derived_cfg["loan_percent_income_computed"]["columns"]
            precision: int = derived_cfg["loan_percent_income_computed"].get(
                "precision", 4
            )
            if cols[0] in df.columns and cols[1] in df.columns:
                df["loan_percent_income_computed"] = np.where(
                    df[cols[1]] > 0, (df[cols[0]] / df[cols[1]]).round(precision), 0.0
                )

        # Check for missing required base features
        required_features: set = set(num_cols + cat_cols)
        existing_features: set = set(df.columns)
        missing_features: set = required_features - existing_features

        if missing_features:
            raise ValueError(
                f"[INPUT ERROR] Input data is missing required features: {missing_features}"
            )

        return df

    def _determine_decision(self, credit_score: float) -> Tuple[str, str]:
        """Applies business cut-off thresholds from configuration settings.

        Args:
            credit_score (float): Calculated applicant credit score.

        Returns:
            Tuple[str, str]: Automated decision outcome ('APPROVED', 'MANUAL_REVIEW', 'REJECTED')
                and corresponding risk band label ('Low Risk', 'Medium Risk', 'High Risk').
        """
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
        """Calculates attribute-level point attribution for waterfall diagnostic charts.

        Args:
            df_clean (pd.DataFrame): Cleaned features DataFrame.
            df_bins (Optional[pd.DataFrame]): Binned features DataFrame matching scorecard specs.

        Returns:
            List[Dict[str, int]]: List of point attribution mappings per record.
        """
        scaler: Any = self.score_scaler
        breakdowns: List[Dict[str, int]] = []
        df_target: pd.DataFrame = df_bins if df_bins is not None else df_clean

        # Extract scaling mapping dictionaries and edges from scaler object
        points_map: Dict[str, Dict[str, int]] = getattr(scaler, "points_map", {})
        bin_edges: Dict[str, Any] = getattr(scaler, "bin_edges", {})

        # Parse string-formatted scorecard table to DataFrame if present
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

                # Strategy 1: Map directly using points_map
                if col in points_map and isinstance(points_map[col], dict):
                    feature_map: Dict[str, int] = points_map[col]

                    # Sub-case A: Direct match on categorical value (e.g., 'RENT', 'OWN')
                    if str(val_clean) in feature_map:
                        points = feature_map[str(val_clean)]
                    # Sub-case B: Direct match on structural bin label (e.g., 'Bin_0')
                    elif val_bin in feature_map:
                        points = feature_map[val_bin]
                    # Sub-case C: Continuous numeric binning lookup using bin_edges
                    elif col in bin_edges:
                        edges = bin_edges[col]
                        if isinstance(edges, (list, np.ndarray)) and isinstance(
                            val_clean, (int, float)
                        ):
                            # Find matching bin index for continuous numeric values
                            bin_idx: int = int(
                                np.digitize([val_clean], edges, right=False)[0] - 1
                            )
                            bin_idx = max(
                                0, min(bin_idx, len(edges) - 2)
                            )  # Clip index within bounds
                            bin_key: str = f"Bin_{bin_idx}"
                            if bin_key in feature_map:
                                points = feature_map[bin_key]

                # Strategy 2: Fallback lookup using scorecard DataFrame
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
        """Executes the end-to-end scoring pipeline for single or batch applicant records.

        Args:
            input_df (pd.DataFrame): Raw applicant feature dataframe.

        Returns:
            Union[Dict[str, Any], pd.DataFrame]: Dictionary containing single record scoring
                results (including score breakdown), or DataFrame containing batch results.
        """
        # Step 1: Input Validation & Feature Derivation
        df_validated: pd.DataFrame = self._validate_and_preprocess_input(input_df)

        # Step 2: Multi-Tier Feature Transformations
        df_clean: pd.DataFrame = self.cleaner.transform(df_validated)
        df_woe: pd.DataFrame = self.woe_transformer.transform(df_clean)
        df_bins: pd.DataFrame = extract_structural_bins(df_clean, self.woe_transformer)

        # Step 3: Predictive Computations
        credit_scores: pd.Series = self.score_scaler.transform(df_bins)
        probabilities: np.ndarray = self.model.predict_probability(df_woe)

        # Step 4: Format Results Payload
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
        print("🚀 [SUCCESS] Inference Pipeline Engine initialized successfully!")
    except Exception as e:
        print(f"❌ [ERROR] Initialization Failed: {e}")
