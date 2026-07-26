"""
Feature Selection Module for Credit Scoring
-------------------------------------------
This module implements a dynamic feature selection pipeline:
1. IV Screening: Drops features with low Information Value (IV) or zero variance.
2. Business Exclusion: Removes features not available at the time of decision.
3. LASSO Regression CV: Selects the most robust predictors using L1 regularization
   and the optimal `lambda.1se` rule.
"""

import warnings
import numpy as np
import pandas as pd
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
    and retain only highly predictive variables for Credit Scoring models.
    """

    def __init__(self, selection_config: Dict[str, Any]) -> None:
        """
        Initializes the CreditFeatureSelector with configuration parameters.

        Args:
            selection_config (Dict[str, Any]): Dictionary containing configuration keys:
                - iv_threshold (float): Minimum Information Value to retain a feature.
                - lasso_cv_splits (int): Number of folds for Cross-Validation.
                - random_state (int): Seed for reproducibility.
                - business_exclude (List[str]): Features to drop for business reasons.
                - lasso_max_iter (int): Maximum iterations for LASSO convergence.
                - lasso_solver (str): Solver for LogisticRegression (e.g., 'saga').
                - penalty (str): Regularization type ('l1' or 'elasticnet').
                - l1_ratio (float): ElasticNet mixing parameter (0 <= l1_ratio <= 1).

        Raises:
            ValueError: If an unsupported penalty type is provided.
        """
        self.iv_threshold: float = selection_config.get("iv_threshold", 0.02)
        self.cv_splits: int = selection_config.get("lasso_cv_splits", 10)
        self.random_state: int = selection_config.get("random_state", 42)
        self.business_exclude: List[str] = selection_config.get("business_exclude", [])

        self.lasso_max_iter: int = selection_config.get("lasso_max_iter", 10000)
        self.lasso_solver: str = selection_config.get("lasso_solver", "saga")

        # Read the penalty parameter with a protective guardrail
        self.penalty: str = selection_config.get("penalty", "l1").lower()
        self.l1_ratio: float = selection_config.get("l1_ratio", 1.0)

        if self.penalty not in ["l1", "elasticnet"]:
            raise ValueError(
                f"[CRITICAL] Penalty '{self.penalty}' is not supported for Feature Selection. "
                "Use 'l1' or 'elasticnet'."
            )

        self.selected_features_: List[str] = []
        self.audit_report_: Dict[str, Any] = {}

    def _make_lasso_pipeline(self, C: float) -> Pipeline:
        """
        Creates a scikit-learn Pipeline with a StandardScaler and LogisticRegression.

        Args:
            C (float): Inverse of regularization strength. Smaller values specify stronger regularization.

        Returns:
            Pipeline: Configured scikit-learn pipeline.
        """
        # Set l1_ratio dynamically (LogisticRegression only accepts l1_ratio if penalty='elasticnet')
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
    ) -> pd.DataFrame:
        """
        Applies IV screening and LASSO Cross-Validation to select the optimal subset of features.
        Explicitly tracks and logs features dropped at each stage for MLOps auditing.

        Args:
            X_woe (pd.DataFrame): Training data encoded with Weight of Evidence (WOE) values.
            y (pd.Series): Target labels (binary: 0 or 1).
            iv_scores (Dict[str, float]): Information Value mapping (e.g., from WOETransformer).

        Returns:
            pd.DataFrame: A sub-setted DataFrame containing only the selected final features.

        Raises:
            ValueError: If no features pass the initial IV and business screening.
        """
        print("\n" + "=" * 80)
        print("🎯 [FEATURE SELECTION] INITIATING DYNAMIC SELECTION PIPELINE")
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

            # Drop features with low IV or constant WOE values (zero variance)
            if iv < self.iv_threshold or X_woe[feat].nunique(dropna=False) <= 1:
                dropped_iv.append(feat)
                continue

            candidate_features.append(feat)

        if not candidate_features:
            raise ValueError(
                "[CRITICAL] No features passed the initial IV and business screening checks."
            )

        X_candidates: pd.DataFrame = X_woe[candidate_features].copy()
        y_array: np.ndarray = y.to_numpy()

        print(
            f"  -> Phase 1: {len(candidate_features)} features passed IV (>={self.iv_threshold}) and business screening."
        )
        print(f"     * [AUDIT] Dropped by Business Rules : {dropped_business}")
        print(f"     * [AUDIT] Dropped by IV / Variance  : {dropped_iv}")

        # ----------------------------------------------------------------------
        # PHASE 2: LASSO CROSS-VALIDATION (lambda.1se optimization)
        # ----------------------------------------------------------------------
        Cs: np.ndarray = np.logspace(-4, 2, 35)
        cv = StratifiedKFold(
            n_splits=self.cv_splits, shuffle=True, random_state=self.random_state
        )
        auc_matrix: np.ndarray = np.zeros((len(Cs), self.cv_splits))

        print(
            f"\n  -> Phase 2: Running LASSO Logistic Regression with {self.cv_splits}-Fold CV..."
        )

        for fold_idx, (train_idx, val_idx) in enumerate(
            cv.split(X_candidates, y_array)
        ):
            X_train_fold: pd.DataFrame = X_candidates.iloc[train_idx]
            y_train_fold: np.ndarray = y_array[train_idx]
            X_val_fold: pd.DataFrame = X_candidates.iloc[val_idx]
            y_val_fold: np.ndarray = y_array[val_idx]

            for c_idx, C in enumerate(Cs):
                pipe = self._make_lasso_pipeline(C)

                with warnings.catch_warnings():
                    # Suppress future/user warnings while keeping ConvergenceWarning active
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    pipe.fit(X_train_fold, y_train_fold)

                # Assuming binary classification; predicting probabilities for the positive class (1)
                probs: np.ndarray = pipe.predict_proba(X_val_fold)[:, 1]
                auc_matrix[c_idx, fold_idx] = roc_auc_score(y_val_fold, probs)

        mean_auc: np.ndarray = np.mean(auc_matrix, axis=1)
        se_auc: np.ndarray = np.std(auc_matrix, axis=1, ddof=1) / np.sqrt(
            self.cv_splits
        )

        best_idx: int = int(np.argmax(mean_auc))
        threshold_1se: float = mean_auc[best_idx] - se_auc[best_idx]

        # Find the smallest C (strongest penalty / largest lambda) that satisfies the 1-SE rule.
        # Since 'Cs' is sorted ascendingly, the first match provides the strongest penalty.
        C_1se: Optional[float] = None
        for c_val, auc_val in zip(Cs, mean_auc):
            if auc_val >= threshold_1se:
                C_1se = c_val
                break

        if C_1se is None:
            C_1se = Cs[best_idx]  # Fallback to lambda.min if the 1-SE logic fails

        # ----------------------------------------------------------------------
        # PHASE 3: FINAL FEATURE EXTRACTION
        # ----------------------------------------------------------------------
        final_pipe = self._make_lasso_pipeline(C_1se)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            final_pipe.fit(X_candidates, y_array)

        coefs: np.ndarray = final_pipe.named_steps["lasso"].coef_[0]

        self.selected_features_ = [
            feat for feat, coef in zip(candidate_features, coefs) if abs(coef) > 1e-8
        ]

        dropped_lasso: List[str] = [
            feat for feat in candidate_features if feat not in self.selected_features_
        ]

        # Failsafe: if LASSO regularization is too aggressive and drops all features
        if not self.selected_features_:
            print(
                "  -> [WARNING] LASSO dropped all features. Falling back to IV candidates."
            )
            self.selected_features_ = candidate_features
            dropped_lasso = []

        print(
            f"\n  -> Phase 3: LASSO lambda.1se selected {len(self.selected_features_)} optimal features."
        )
        print(f"     * [AUDIT] Dropped by LASSO (Beta=0) : {dropped_lasso}")
        print(f"     * Final Features Retained           : {self.selected_features_}")

        # ----------------------------------------------------------------------
        # MLOPS AUDIT REGISTRY
        # ----------------------------------------------------------------------
        self.audit_report_ = {
            "initial_features": list(X_woe.columns),
            "business_excluded_features": dropped_business,
            "iv_dropped_features": dropped_iv,
            "iv_qualified_features": candidate_features,
            "lasso_dropped_features": dropped_lasso,
            "lasso_selected_features": self.selected_features_,
            "lasso_c_1se_value": float(C_1se),
        }

        return X_woe[self.selected_features_].copy()

    def transform(self, X_woe: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the DataFrame to keep only the features successfully selected during fit_transform.

        Args:
            X_woe (pd.DataFrame): Inference data encoded with WOE values.

        Returns:
            pd.DataFrame: A sub-setted DataFrame containing only the selected features.

        Raises:
            KeyError: If any of the required selected features are missing from the input dataset.
        """
        missing_cols: List[str] = [
            col for col in self.selected_features_ if col not in X_woe.columns
        ]
        if missing_cols:
            raise KeyError(
                f"[CRITICAL] Missing selected features in validation/test dataset: {missing_cols}"
            )

        return X_woe[self.selected_features_].copy()
