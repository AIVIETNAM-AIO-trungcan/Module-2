"""
Feature Selection Module for Credit Scoring
-------------------------------------------
This module implements a dynamic dual-branch feature selection pipeline:
1. IV & Business Screening: Drops uninformative (low IV), suspicious (high IV leakage),
   and business-excluded features.
2. VIF Screening: Iteratively removes multicollinear features to form the "Baseline" subset.
3. LASSO Regression CV: Applies the optimal `lambda.1se` rule to extract the "Best Subset"
   from the Baseline features.

*Outputs both Baseline and Best Subset features for Champion vs. Challenger model benchmarking.*
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Dict, Any, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning


class CreditFeatureSelector:
    """
    Executes automated feature selection to prevent multicollinearity
    and retain highly predictive variables. Supports dual-branch output
    (Baseline vs. Best Subset) for downstream model benchmarking.
    """

    def __init__(self, selection_config: Dict[str, Any]) -> None:
        """
        Initializes the CreditFeatureSelector with configuration parameters.

        Args:
            selection_config (Dict[str, Any]): Dictionary containing feature selection rules.
        """
        # Tier 1: Baseline Filters
        self.min_iv: float = selection_config.get("min_iv_threshold", 0.02)
        self.max_iv: float = selection_config.get("max_iv_threshold", 0.50)
        self.vif_threshold: float = selection_config.get("vif_threshold", 5.0)
        self.business_exclude: List[str] = selection_config.get("business_exclude", [])

        # Tier 2: Best Subset (LASSO) Filters
        subset_cfg: Dict[str, Any] = selection_config.get("best_subset", {})
        self.use_best_subset: bool = subset_cfg.get("enabled", True)
        self.cv_splits: int = subset_cfg.get("lasso_cv_splits", 10)
        self.random_state: int = subset_cfg.get("random_state", 42)
        self.lasso_max_iter: int = subset_cfg.get("lasso_max_iter", 10000)
        self.lasso_solver: str = subset_cfg.get("lasso_solver", "saga")
        self.penalty: str = subset_cfg.get("penalty", "l1").lower()
        self.l1_ratio: float = subset_cfg.get("l1_ratio", 1.0)

        if self.penalty not in ["l1", "elasticnet"]:
            raise ValueError(
                f"[CRITICAL] Penalty '{self.penalty}' is not supported for Feature Selection. "
                "Use 'l1' or 'elasticnet'."
            )

        # Registries
        self.baseline_features_: List[str] = []
        self.subset_features_: List[str] = []
        self.audit_report_: Dict[str, Any] = {}

    def _apply_vif_screening(self, X: pd.DataFrame, features: List[str]) -> List[str]:
        """
        Iteratively calculates Variance Inflation Factor (VIF) and removes the feature
        with the highest VIF until all features are below the threshold.
        """
        retained_features = features.copy()

        while True:
            if not retained_features:
                break

            X_eval = X[retained_features].astype(float)

            # Bypass statsmodels.api to avoid environment dependency crashes
            X_with_const = X_eval.copy()
            X_with_const.insert(0, "const", 1.0)

            vif_data = []
            cols = X_with_const.columns

            for i in range(X_with_const.shape[1]):
                if cols[i] == "const":
                    continue
                try:
                    # Suppress invalid value warnings from statsmodels internally
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        vif_val = variance_inflation_factor(X_with_const.values, i)
                except Exception:
                    vif_val = np.inf
                vif_data.append((cols[i], vif_val))

            if not vif_data:
                break

            vif_df = pd.DataFrame(vif_data, columns=["Feature", "VIF"])
            max_vif_row = vif_df.loc[vif_df["VIF"].idxmax()]

            if max_vif_row["VIF"] > self.vif_threshold:
                retained_features.remove(max_vif_row["Feature"])
            else:
                break

        return retained_features

    def _make_lasso_pipeline(self, C: float) -> Pipeline:
        """
        Creates a scikit-learn Pipeline with a StandardScaler and LogisticRegression.
        """
        l1_ratio_val: Optional[float] = (
            self.l1_ratio if self.penalty == "elasticnet" else None
        )

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lasso",
                    LogisticRegression(
                        penalty=self.penalty,
                        solver=self.lasso_solver,
                        C=C,
                        l1_ratio=l1_ratio_val,
                        max_iter=self.lasso_max_iter,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit_transform(
        self, X_woe: pd.DataFrame, y: pd.Series, iv_scores: Dict[str, float]
    ) -> Dict[str, pd.DataFrame]:
        """
        Executes IV, Business, VIF, and LASSO screening to generate Baseline and Best Subset features.

        Args:
            X_woe (pd.DataFrame): Training data encoded with Weight of Evidence (WOE) values.
            y (pd.Series): Target labels (binary: 0 or 1).
            iv_scores (Dict[str, float]): Information Value mapping.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing 'baseline' and 'subset' DataFrames.
        """
        print("\n" + "=" * 80)
        print("🎯 [FEATURE SELECTION] INITIATING DUAL-BRANCH SELECTION PIPELINE")
        print("=" * 80)

        # ----------------------------------------------------------------------
        # PHASE 1: IV SCREENING & BUSINESS EXCLUSION
        # ----------------------------------------------------------------------
        candidate_features: List[str] = []
        dropped_business: List[str] = []
        dropped_iv: List[str] = []

        for feat, iv in iv_scores.items():
            if feat in self.business_exclude:
                dropped_business.append(feat)
                continue

            if (
                iv < self.min_iv
                or iv > self.max_iv
                or X_woe[feat].nunique(dropna=False) <= 1
            ):
                dropped_iv.append(feat)
                continue

            candidate_features.append(feat)

        if not candidate_features:
            raise ValueError(
                "[CRITICAL] No features passed the initial IV/Business screening."
            )

        # ----------------------------------------------------------------------
        # PHASE 2: VIF SCREENING (MULTICOLLINEARITY) -> YIELDS BASELINE FEATURES
        # ----------------------------------------------------------------------
        print(
            f"  -> Phase 1 & 2: Running VIF Screening (Threshold <= {self.vif_threshold})..."
        )
        self.baseline_features_ = self._apply_vif_screening(X_woe, candidate_features)

        dropped_vif = [
            f for f in candidate_features if f not in self.baseline_features_
        ]

        print(f"     * [AUDIT] Dropped by Business Rules : {dropped_business}")
        print(f"     * [AUDIT] Dropped by IV Thresholds  : {dropped_iv}")
        print(f"     * [AUDIT] Dropped by VIF (> {self.vif_threshold}) : {dropped_vif}")
        print(
            f"     * [BASELINE] Features Retained      : {len(self.baseline_features_)}"
        )

        X_baseline: pd.DataFrame = X_woe[self.baseline_features_].copy()
        y_array: np.ndarray = y.to_numpy()

        # ----------------------------------------------------------------------
        # PHASE 3: LASSO CROSS-VALIDATION -> YIELDS BEST SUBSET FEATURES
        # ----------------------------------------------------------------------
        self.subset_features_ = self.baseline_features_.copy()
        dropped_lasso: List[str] = []
        C_1se: float = 1.0

        if self.use_best_subset:
            Cs: np.ndarray = np.logspace(-4, 2, 35)
            cv = StratifiedKFold(
                n_splits=self.cv_splits, shuffle=True, random_state=self.random_state
            )
            auc_matrix: np.ndarray = np.zeros((len(Cs), self.cv_splits))

            print(
                f"\n  -> Phase 3: Running LASSO Challenger with {self.cv_splits}-Fold CV (1-SE Rule)..."
            )

            for fold_idx, (train_idx, val_idx) in enumerate(
                cv.split(X_baseline, y_array)
            ):
                X_train_fold, y_train_fold = (
                    X_baseline.iloc[train_idx],
                    y_array[train_idx],
                )
                X_val_fold, y_val_fold = X_baseline.iloc[val_idx], y_array[val_idx]

                for c_idx, C in enumerate(Cs):
                    pipe = self._make_lasso_pipeline(C)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        warnings.filterwarnings("ignore", category=UserWarning)
                        pipe.fit(X_train_fold, y_train_fold)

                    probs: np.ndarray = pipe.predict_proba(X_val_fold)[:, 1]
                    auc_matrix[c_idx, fold_idx] = roc_auc_score(y_val_fold, probs)

            mean_auc: np.ndarray = np.mean(auc_matrix, axis=1)
            se_auc: np.ndarray = np.std(auc_matrix, axis=1, ddof=1) / np.sqrt(
                self.cv_splits
            )

            best_idx: int = int(np.argmax(mean_auc))
            threshold_1se: float = mean_auc[best_idx] - se_auc[best_idx]

            C_1se_val: Optional[float] = None
            for c_val, auc_val in zip(Cs, mean_auc):
                if auc_val >= threshold_1se:
                    C_1se_val = c_val
                    break

            C_1se = C_1se_val if C_1se_val is not None else Cs[best_idx]

            # Extract Final Coefficients
            final_pipe = self._make_lasso_pipeline(C_1se)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                final_pipe.fit(X_baseline, y_array)

            coefs: np.ndarray = final_pipe.named_steps["lasso"].coef_[0]
            self.subset_features_ = [
                feat
                for feat, coef in zip(self.baseline_features_, coefs)
                if abs(coef) > 1e-8
            ]
            dropped_lasso = [
                f for f in self.baseline_features_ if f not in self.subset_features_
            ]

            if not self.subset_features_:
                print(
                    "  -> [WARNING] LASSO dropped all features. Falling back to Baseline candidates."
                )
                self.subset_features_ = self.baseline_features_.copy()
                dropped_lasso = []

            print(f"     * [AUDIT] Dropped by LASSO (Beta=0) : {dropped_lasso}")
            print(
                f"     * [CHALLENGER] Best Subset Retained : {len(self.subset_features_)}"
            )

        # ----------------------------------------------------------------------
        # MLOPS AUDIT REGISTRY
        # ----------------------------------------------------------------------
        self.audit_report_ = {
            "initial_features": list(X_woe.columns),
            "business_excluded_features": dropped_business,
            "iv_dropped_features": dropped_iv,
            "vif_dropped_features": dropped_vif,
            "baseline_features": self.baseline_features_,
            "lasso_dropped_features": dropped_lasso,
            "subset_features": self.subset_features_,
            "lasso_c_1se_value": float(C_1se),
        }

        return {
            "baseline": X_woe[self.baseline_features_].copy(),
            "subset": X_woe[self.subset_features_].copy(),
        }

    def transform(self, X_woe: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Filters the DataFrame into the Baseline and Best Subset feature lists.

        Args:
            X_woe (pd.DataFrame): Inference data encoded with WOE values.

        Returns:
            Dict[str, pd.DataFrame]: Sub-setted DataFrames matching the dual-branch output.
        """
        missing_base = [
            col for col in self.baseline_features_ if col not in X_woe.columns
        ]
        missing_sub = [col for col in self.subset_features_ if col not in X_woe.columns]

        if missing_base or missing_sub:
            raise KeyError(
                f"[CRITICAL] Missing selected features in validation dataset."
            )

        return {
            "baseline": X_woe[self.baseline_features_].copy(),
            "subset": X_woe[self.subset_features_].copy(),
        }
