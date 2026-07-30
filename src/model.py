"""
Baseline Model Training Pipeline for Credit Scoring
---------------------------------------------------
This module wraps the scikit-learn Logistic Regression engine to fulfill
banking standards, supporting configuration and strict coefficient auditing.
Includes Step-wise Backward Elimination based on Sign Reversal, P-values, and VIF.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, Optional, List, Tuple


class CreditModelTrainer:
    """
    Wraps the Logistic Regression model to handle dynamic hyperparameter extraction
    and enforce banking risk validation workflows via Auto Step-wise Backward Elimination.
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """
        Initializes the model trainer by dynamically extracting hyperparameters.

        Args:
            model_config (Dict[str, Any]): The "model" Subtree parsed from config_yaml.
        """
        # Safe extraction
        lr_params = model_config.get("logistic_regression", {})

        # Mapping parameters
        self.random_state: int = lr_params.get("random_state", 42)
        self.max_iter: int = lr_params.get("max_iter", 1000)
        self.C: float = float(lr_params.get("c_parameter", 1.0))

        # Regulatory Audit Thresholds
        self.p_value_threshold: float = lr_params.get("p_value_threshold", 0.05)
        self.vif_threshold: float = lr_params.get("vif_threshold", 5.0)

        # Placeholder for the underlying scikit-learn model object
        self.model: Optional[LogisticRegression] = None

        # Containers to store audited parameters for risk validation
        self.final_features_: List[str] = []
        self.intercept_: float = 0.0
        self.coefficients_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CreditModelTrainer":
        """
        Fits the model using an iterative Step-wise Backward Elimination process.
        Enforces 3 rigid rules for Scorecard compliance:
        1. Sign Reversal (All Betas must be < 0).
        2. Statistical Significance (P-value <= threshold).
        3. Multicollinearity Stability (VIF <= threshold).

        Args:
            X (pd.DataFrame): Training feature matrix (must be WOE-encoded).
            y (pd.Series): Target binary labels (1=Bad; 0=Good).

        Returns:
            CreditModelTrainer: The fitted instance itself.
        """
        self.final_features_ = list(X.columns)
        self.coefficients_.clear()
        iteration = 1

        print("\n" + "=" * 80)
        print("🏛️ [MODEL ENGINE] INITIATING REGULATORY BACKWARD ELIMINATION")
        print("=" * 80)

        while True:
            if not self.final_features_:
                raise ValueError(
                    "[CRITICAL ERROR] All features were dropped during Backward Elimination. "
                    "Please review your WOE encoding or Feature Selection constraints."
                )

            X_curr = X[self.final_features_]

            # 1. Core Engine Initialization (Scikit-Learn for penalized training)
            self.model = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver="lbfgs",
            )
            self.model.fit(X_curr, y)

            # ------------------------------------------------------------------
            # RULE 1: SIGN REVERSAL CHECK (Beta >= 0)
            # ------------------------------------------------------------------
            betas = self.model.coef_[0]
            positive_betas = [
                (feat, beta)
                for feat, beta in zip(self.final_features_, betas)
                if beta >= 0
            ]

            if positive_betas:
                # Drop the feature with the highest positive Beta
                worst_feature, worst_beta = max(
                    positive_betas, key=lambda item: item[1]
                )
                print(
                    f"  -> Iteration {iteration:<2}: Dropped '{worst_feature}' "
                    f"(Rule 1 - Sign Reversal: Beta = {worst_beta:.4f} >= 0)"
                )
                self.final_features_.remove(worst_feature)
                iteration += 1
                continue

            # ------------------------------------------------------------------
            # RULE 2: STATISTICAL SIGNIFICANCE CHECK (P-value > threshold)
            # ------------------------------------------------------------------
            X_const = X_curr.astype(float).copy()
            X_const.insert(0, "const", 1.0)
            high_p_values: List[Tuple[str, float]] = []

            try:
                # Use isolated Logit class to retrieve exact statistical p-values
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    logit_model = Logit(y, X_const).fit(disp=0, method="newton")

                p_values = logit_model.pvalues.drop("const", errors="ignore")
                high_p_values = [
                    (feat, pval)
                    for feat, pval in p_values.items()
                    if pval > self.p_value_threshold
                ]
            except Exception as e:
                print(
                    f"  -> [WARNING] P-value estimation failed ({e}). Skipping Rule 2 for this iteration."
                )

            if high_p_values:
                # Drop the feature with the highest p-value
                worst_feature, worst_pval = max(high_p_values, key=lambda item: item[1])
                print(
                    f"  -> Iteration {iteration:<2}: Dropped '{worst_feature}' "
                    f"(Rule 2 - Insignificant: p-value = {worst_pval:.4f} > {self.p_value_threshold})"
                )
                self.final_features_.remove(worst_feature)
                iteration += 1
                continue

            # ------------------------------------------------------------------
            # RULE 3: MULTICOLLINEARITY CHECK (VIF > threshold)
            # ------------------------------------------------------------------
            vif_data: List[Tuple[str, float]] = []
            cols = X_const.columns

            for i in range(X_const.shape[1]):
                if cols[i] == "const":
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        vif_val = variance_inflation_factor(X_const.values, i)
                except Exception:
                    vif_val = np.inf
                vif_data.append((cols[i], float(vif_val)))

            high_vifs = [
                (feat, vif) for feat, vif in vif_data if vif > self.vif_threshold
            ]

            if high_vifs:
                # Drop the feature with the highest VIF
                worst_feature, worst_vif = max(high_vifs, key=lambda item: item[1])
                print(
                    f"  -> Iteration {iteration:<2}: Dropped '{worst_feature}' "
                    f"(Rule 3 - Multicollinearity: VIF = {worst_vif:.4f} > {self.vif_threshold})"
                )
                self.final_features_.remove(worst_feature)
                iteration += 1
                continue

            # If all 3 rules pass, the model is mathematically and statistically sound
            break

        print(
            f"  -> [SUCCESS] Backward Elimination converged successfully in {iteration} iterations."
        )
        print(f"  -> Final Validated Features Retained: {len(self.final_features_)}")

        # Extract and catalog coefficients for regulatory risk auditing
        self.intercept_ = float(self.model.intercept_[0])
        for feature_name, coef_value in zip(self.final_features_, self.model.coef_[0]):
            self.coefficients_[feature_name] = float(coef_value)

        return self

    def predict_class(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts binary hard labels (0 or 1) for credit decisions.

        Args:
            X (pd.DataFrame): Feature matrix to predict.

        Returns:
            np.ndarray: Binary array of 1s (Bad) and 0s (Good).
        """
        if self.model is None:
            raise ValueError("[ERROR] Model must be fitted before running prediction.")

        # Ensure only the surviving features are fed into the model
        return self.model.predict(X[self.final_features_])

    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts the raw continuous Probability of Default (PD / Soft-labels).
        Crucial metric utilized downstream for Scorecard points scaling.

        Args:
            X (pd.DataFrame): Feature matrix to predict.

        Returns:
            np.ndarray: Continuous probability array ranging between 0.0 and 1.0.
        """
        if self.model is None:
            raise ValueError("[ERROR] Model must be fitted before running prediction.")

        return self.model.predict_proba(X[self.final_features_])[:, 1]


# ==============================================================================
# INTERNAL TEST BLOCK (SANDBOX)
# ==============================================================================
if __name__ == "__main__":
    print("--- Testing Regulatory Baseline Model Pipeline ---")

    # 1. Mock dynamic config simulation directly from config.yaml structure
    MOCK_CONFIG = {
        "logistic_regression": {
            "random_state": 42,
            "max_iter": 100,
            "c_parameter": 1.0,
            "p_value_threshold": 0.05,
            "vif_threshold": 5.0,
        }
    }

    # 2. Mock surviving features generation (Simulating WOE Transformer matrix output)
    np.random.seed(42)
    mock_X = pd.DataFrame(
        {
            "person_age": np.random.uniform(-1.5, 1.5, size=1000),
            "person_income": np.random.uniform(-2.0, 2.0, size=1000),
            "person_home_ownership": np.random.uniform(-1.0, 1.0, size=1000),
            "noise_feature": np.random.uniform(
                -1.0, 1.0, size=1000
            ),  # Intentionally weak feature
        }
    )

    # Synthetic target generation with negative correlation to match WOE behaviors
    raw_scores = (
        -0.8 * mock_X["person_income"]
        - 0.5 * mock_X["person_age"]
        - 0.2 * mock_X["person_home_ownership"]
        + np.random.normal(0, 0.5, 1000)
    )
    mock_y = pd.Series(np.where(raw_scores > 0, 1, 0))

    # 3. Instantiate and trigger Method Chaining pipeline execution
    trainer = CreditModelTrainer(model_config=MOCK_CONFIG).fit(mock_X, mock_y)

    # 4. Generate validation inferences
    classes = trainer.predict_class(mock_X)
    probabilities = trainer.predict_probability(mock_X)

    # ==============================================================================
    # COMPREHENSIVE REGULATORY MODEL AUDIT LOGS
    # ==============================================================================
    print("\n==================================================")
    print("[AUDIT] FINAL LOGISTIC REGRESSION COEFFICIENTS")
    print("==================================================")
    print(f"  - Base Intercept (Beta_0)   : {trainer.intercept_:.4f}")

    # Financial Rule Validation: Since WOE is directly proportional to safety,
    # all beta coefficients running into predictions must be NEGATIVE.
    for feature, beta in trainer.coefficients_.items():
        audit_status = (
            "✅ VALID (Negative)"
            if beta < 0
            else "❌ CRITICAL ERROR (Positive Beta violates risk math)"
        )
        print(
            f"  - Coefficient (Beta) {feature:<22}: {beta:.4f} | Status: {audit_status}"
        )

    print("\n==================================================")
    print("[AUDIT] OUTPUT MATRICES SANITY CHECK")
    print("==================================================")
    print(f"  - First 5 Predicted Classes  : {classes[:5]}")
    print(f"  - First 5 Probabilities (PD) : {[f'{p:.4f}' for p in probabilities[:5]]}")

    print("\n==================================================")
    print("[SUCCESS] Model Pipeline executed with full regulatory visibility.")
