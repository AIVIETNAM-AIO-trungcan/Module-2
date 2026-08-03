"""
Feature Selection Module for Credit Scoring
-------------------------------------------
This module implements a dynamic tri-branch feature selection pipeline
to match the candidate evaluation strategy from Notebook 4:

1. Model 1 (IV-screened full): Retains all features passing IV screening (Benchmark).
2. Intermediate (Pre-decision full financial): Removes post-decision features (e.g. loan_grade) to prevent data leakage.
3. Model 2 (Explainable pre-decision): Removes redundant financial variables, keeping only the computed ratio.
"""

import pandas as pd
from typing import List, Dict, Any


class CreditFeatureSelector:
    """
    Executes an explainable, rule-based feature selection process.
    Supports tri-branch output (Model 1, Intermediate, Model 2) for downstream CV benchmarking.
    """

    def __init__(self, selection_config: Dict[str, Any]) -> None:
        """
        Initializes the CreditFeatureSelector with configuration parameters.

        Args:
            selection_config (Dict[str, Any]): Dictionary containing feature selection rules.
        """
        # Tier 1: Baseline Filters (IV Screening)
        self.min_iv: float = selection_config.get("min_iv_threshold", 0.02)
        self.max_iv: float = selection_config.get("max_iv_threshold", 1.00)

        # Tier 2: Explainable Subset Rules (Business & Redundancy Exclusion)
        explainable_cfg: Dict[str, Any] = selection_config.get("explainable_rules", {})
        self.use_explainable_rules: bool = explainable_cfg.get("enabled", True)
        self.post_decision_features: List[str] = explainable_cfg.get(
            "post_decision_features", []
        )
        self.redundancy_drop: List[str] = explainable_cfg.get("redundancy_drop", [])
        self.redundancy_keep: str = explainable_cfg.get("redundancy_keep", "")

        # Registries
        self.baseline_features_: List[str] = []
        self.candidate_feature_sets_: Dict[str, List[str]] = {}
        self.audit_report_: Dict[str, Any] = {}

    def get_candidate_feature_sets(self) -> Dict[str, List[str]]:
        """
        Generates the 3 distinct feature sets based on business rules for Model Comparison.
        Must be called after IV screening has populated self.baseline_features_.
        """
        if not self.baseline_features_:
            raise ValueError(
                "[ERROR] Must fit the selector before generating candidate sets."
            )

        # 1. Model 1 - Benchmark Full (No rules applied, only IV screened)
        model_1_features = self.baseline_features_.copy()

        # 2. Intermediate Model - Pre-decision (Removes data leakage features like loan_grade)
        model_intermediate_features = [
            f for f in model_1_features if f not in self.post_decision_features
        ]

        # 3. Model 2 - Explainable (Removes redundant financial variables like raw income/loan_amnt)
        model_2_features = [
            f for f in model_intermediate_features if f not in self.redundancy_drop
        ]

        self.candidate_feature_sets_ = {
            "model_1_iv_screened_full": model_1_features,
            "model_intermediate_predecision_full_financial": model_intermediate_features,
            "model_2_explainable_predecision": model_2_features,
        }

        return self.candidate_feature_sets_

    def fit_transform(
        self, X_woe: pd.DataFrame, y: pd.Series, iv_scores: Dict[str, float]
    ) -> Dict[str, pd.DataFrame]:
        """
        Executes IV screening and rule-based business logic to generate 3 Candidate DataFrames.

        Args:
            X_woe (pd.DataFrame): Training data encoded with Weight of Evidence (WOE) values.
            y (pd.Series): Target labels (binary: 0 or 1).
            iv_scores (Dict[str, float]): Information Value mapping from OptBinning.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing dataframes for 'model_1', 'intermediate', and 'model_2'.
        """
        print("\n" + "=" * 80)
        print("🎯 [FEATURE SELECTION] INITIATING EXPLAINABLE TRI-BRANCH PIPELINE")
        print("=" * 80)

        # ----------------------------------------------------------------------
        # PHASE 1: IV SCREENING -> YIELDS BASELINE FEATURES
        # ----------------------------------------------------------------------
        print(
            f"  -> Phase 1: Running IV Screening (Threshold: {self.min_iv} <= IV <= {self.max_iv})..."
        )

        dropped_iv: List[str] = []
        dropped_constant: List[str] = []
        self.baseline_features_ = []

        for feat, iv in iv_scores.items():
            if feat not in X_woe.columns:
                continue

            # Drop if feature has no variance (Constant WOE)
            if X_woe[feat].nunique(dropna=False) <= 1:
                dropped_constant.append(feat)
                continue

            # Drop if feature falls outside IV bounds
            if iv < self.min_iv or iv > self.max_iv:
                dropped_iv.append(feat)
                continue

            self.baseline_features_.append(feat)

        if not self.baseline_features_:
            raise ValueError("[CRITICAL] No features passed the initial IV screening.")

        print(f"     * [AUDIT] Dropped by Constant WOE : {dropped_constant}")
        print(f"     * [AUDIT] Dropped by IV Threshold : {dropped_iv}")
        print(
            f"     * [BASELINE] Features Retained    : {len(self.baseline_features_)}"
        )

        # ----------------------------------------------------------------------
        # PHASE 2: EXPLAINABLE BUSINESS RULES -> YIELDS CANDIDATE SETS
        # ----------------------------------------------------------------------
        if self.use_explainable_rules:
            print(
                f"\n  -> Phase 2: Applying Explainable Business Rules for Pre-Decision Models..."
            )
            candidate_sets = self.get_candidate_feature_sets()

            print(
                f"     * [AUDIT] Dropped Post-Decision (Leakage) : {self.post_decision_features}"
            )
            print(
                f"     * [AUDIT] Dropped Redundant Financial     : {self.redundancy_drop}"
            )
            print(
                f"     * [CANDIDATE 1] Model 1 Retained          : {len(candidate_sets['model_1_iv_screened_full'])} features"
            )
            print(
                f"     * [CANDIDATE 2] Intermediate Retained     : {len(candidate_sets['model_intermediate_predecision_full_financial'])} features"
            )
            print(
                f"     * [CANDIDATE 3] Model 2 Retained          : {len(candidate_sets['model_2_explainable_predecision'])} features"
            )
        else:
            print(
                "\n  -> Phase 2: Explainable Rules disabled. All branches identical to Baseline."
            )
            base_set = self.baseline_features_.copy()
            self.candidate_feature_sets_ = {
                "model_1_iv_screened_full": base_set,
                "model_intermediate_predecision_full_financial": base_set,
                "model_2_explainable_predecision": base_set,
            }

        # ----------------------------------------------------------------------
        # MLOPS AUDIT REGISTRY
        # ----------------------------------------------------------------------
        self.audit_report_ = {
            "initial_features": list(X_woe.columns),
            "iv_dropped_features": dropped_iv,
            "constant_woe_dropped": dropped_constant,
            "baseline_features": self.baseline_features_,
            "post_decision_dropped": (
                self.post_decision_features if self.use_explainable_rules else []
            ),
            "redundancy_dropped": (
                self.redundancy_drop if self.use_explainable_rules else []
            ),
            "candidate_sets": self.candidate_feature_sets_,
        }

        # Generate the mapped DataFrames for output
        output_dfs = {}
        for model_key, features in self.candidate_feature_sets_.items():
            output_dfs[model_key] = X_woe[features].copy()

        return output_dfs

    def transform(self, X_woe: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Filters the DataFrame into the 3 candidate feature lists.

        Args:
            X_woe (pd.DataFrame): Inference data encoded with WOE values.

        Returns:
            Dict[str, pd.DataFrame]: Sub-setted DataFrames matching the tri-branch output.
        """
        if not self.candidate_feature_sets_:
            raise RuntimeError(
                "[CRITICAL] Selector has not been fitted. Call fit_transform first."
            )

        output_dfs = {}
        for model_key, features in self.candidate_feature_sets_.items():
            missing_cols = [col for col in features if col not in X_woe.columns]
            if missing_cols:
                raise KeyError(
                    f"[CRITICAL] Missing features for {model_key} in validation/test dataset: {missing_cols}"
                )
            output_dfs[model_key] = X_woe[features].copy()

        return output_dfs
