"""
Baseline Model Training Pipeline for Credit Scoring
---------------------------------------------------
This module wraps the scikit-learn Logistic Regression engine to fulfill
banking standards, supporting confguration and strict coefficient auditing.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, Optional


class CreditModelTrainer:
    """
    Wraps the Loogistic Regression model to handle dynamic hyperparameter extraction
    and enforce banking risk validation workflows via Auto Backward Elimination.
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """
        Initializes the model trainer by dynamically extracting hyperparameters.

        Args:
            model_config (Dict[str, Any]): The "model" Subtree parsed from config_yaml
        """
        # Safe extraction
        lr_params = model_config.get("logistic_regression", {})

        # Mapping parameters
        self.random_state: int = lr_params.get("random_state", 42)
        self.max_iter: int = lr_params.get("max_iter", 100)

        # Note: 'c_parameter' in YAML maps directly to 'C' (inverse of regularization strength)
        self.C: float = float(lr_params.get("c_parameter", 1.0))

        # Placeholder for the underlying scikit-learn model object
        self.model: Optional[LogisticRegression] = None

        # Containers to store audited parameters for risk validation
        self.intercept_: float = 0.0
        self.coefficients_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CreditModelTrainer":
        """
        Fit the Logistic Regression model using the pre-extracted dynamic hyperparameters
        Fits the model using an iterative Backward Elimination process to ensure
        all beta coefficients are negative, complying with WOE scorecard math.

        Args:
            X (pd.DataFrame): Training feature matrix (must be WOE-encoded).
            y (pd.Series): Target binary labels (1=Bad; 0=Good).

        Returns:
            CreditModelTrainer: The fitted instance itself.
        """
        self.final_features_ = list(X.columns)
        iteration = 1

        while True:
            # Core Engine Initialization
            self.model = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver="lbfgs",  # Standard stable solver for credit scoring
            )

            # Train using only the currently surviving features
            self.model.fit(X[self.final_features_], y)

            # Check for Sign Reversal (Beta >= 0)
            betas = self.model.coef_[0]
            positive_betas = [
                feat for feat, beta in zip(self.final_features_, betas) if beta >= 0
            ]

            if not positive_betas:
                # All betas are strictly negative. The model is mathematically sound.
                break

            # Audit & Traceability: Log the dropped features due to sign reversal
            print(
                f"[REGULATORY AUDIT] Iteration {iteration}: Safely dropped {len(positive_betas)} features "
                f"due to Sign Reversal (Beta >= 0) -> {positive_betas}"
            )

            # Drop invalid features and loop again for retraining
            for feat in positive_betas:
                self.final_features_.remove(feat)

            iteration += 1

        # Extract and catalog coefficients for regulatory risk auditing
        self.intercept_ = float(self.model.intercept_[0])
        for feature_name, coef_value in zip(self.final_features_, self.model.coef_[0]):
            self.coefficients_[feature_name] = float(coef_value)

        return self

    def predict_class(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts binary hard labels (0 or 1) for credit decisions.

        Args:
            X (pd.DataFrame): Feature matrix to  predict.

        Return:
            np.ndarray: Binary array of 1s (Bad) and 0s (Good)
        """
        if self.model is None:
            raise ValueError("[ERROR] Model must be fitted before running prediction.")
        return self.model.predict(X)

    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts the raw continious Probability of Default (PD / Soft-labels).
        Crucial metric utilized downstream for Scorecard points scaling.

        Args:
            X (pd.DataFrame): Feature matrix to predict.

        Return:
            np.ndarray: Continuous probability array ranging between 0.0 and 1.0.
        """
        if self.model is None:
            raise ValueError("[ERROR] Model must be fitted before running prediction.")
        return self.model.predict_proba(X)[:, 1]


# INTERNAL TEST BLOCK (SANDBOX)

if __name__ == "__main__":
    print("--- Testing Baseline Model Pipeline ---")

    # 1. Mock dynamic config simulation directly from config.yaml structure
    MOCK_CONFIG = {
        "logistic_regression": {"random_state": 42, "max_iter": 100, "c_parameter": 1.0}
    }

    # 2. Mock surviving features generation (Simulating WOE Transformer matrix output)
    np.random.seed(42)
    mock_X = pd.DataFrame(
        {
            "person_age": np.random.uniform(-1.5, 1.5, size=100),
            "person_income": np.random.uniform(-2.0, 2.0, size=100),
            "person_home_ownership": np.random.uniform(-1.0, 1.0, size=100),
        }
    )
    # Synthetic target generation with negative correlation to match WOE behaviors
    raw_scores = (
        -0.8 * mock_X["person_income"]
        - 0.5 * mock_X["person_age"]
        + np.random.normal(0, 0.5, 100)
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
    print("[AUDIT] LOGISTIC REGRESSION COEFFICIENTS VALIDATION")
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
            f"  - Coefficient (Beta) {feature:<10}: {beta:.4f} | Status: {audit_status}"
        )

    print("\n==================================================")
    print("[AUDIT] OUTPUT MATRICES SANITY CHECK")
    print("==================================================")
    print(f"  - First 5 Predicted Classes   : {classes[:5]}")
    print(f"  - First 5 Probabilities (PD) : {[f'{p:.4f}' for p in probabilities[:5]]}")

    print("\n==================================================")
    print("[SUCCESS] Model Pipeline executed with full regulatory visibility.")
