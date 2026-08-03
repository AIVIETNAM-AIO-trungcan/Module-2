"""
Model Training and Evaluation Pipeline for Credit Scoring
---------------------------------------------------------
This module uses the statsmodels GLM (Generalized Linear Model) engine with a
Binomial family (Logistic Regression) to fulfill strict banking audit standards.
It provides 10-Fold Cross-Validation evaluation (for Champion selection) and
final model fitting with comprehensive statistical extraction (P-values, CI, Odds Ratio).
"""

import warnings
import numpy as np
import pandas as pd

# [UPDATED] Direct imports to bypass the statsmodels.api (and 'tsa' module) bug
# caused by pandas deprecation in newer environments.
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families.family import Binomial
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, log_loss
from typing import Dict, Any, Optional, List, Tuple


class CreditModelTrainer:
    """
    Wraps the statsmodels GLM algorithm. Provides OOF Cross-Validation metrics
    and generates detailed statistical audit reports required for Scorecard scaling.
    """

    def __init__(self, model_config: Dict[str, Any] = None) -> None:
        """
        Initializes the model trainer.
        """
        if model_config is None:
            model_config = {}

        lr_params = model_config.get("logistic_regression", {})
        self.vif_threshold: float = lr_params.get("vif_threshold", 5.0)

        self.model: Optional[Any] = None
        self.final_features_: List[str] = []
        self.intercept_: float = 0.0
        self.coefficients_: Dict[str, float] = {}
        self.audit_report_: Dict[str, Any] = {}

    def _calculate_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculates Variance Inflation Factor for multi-collinearity checks."""
        X_const = add_constant(X.astype(float), has_constant="add")
        rows = []
        for i, col in enumerate(X_const.columns):
            if col == "const":
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vif_val = variance_inflation_factor(X_const.values, i)
            except Exception:
                vif_val = np.inf
            rows.append({"Variable": col, "VIF": float(vif_val)})

        return (
            pd.DataFrame(rows)
            .sort_values("VIF", ascending=False)
            .reset_index(drop=True)
        )

    def _calculate_ks(self, y_true: np.ndarray, probability: np.ndarray) -> float:
        """Computes the Kolmogorov-Smirnov (KS) statistic."""
        fpr, tpr, _ = roc_curve(y_true, probability)
        return float(np.max(tpr - fpr))

    def _calculate_calibration(
        self, y_true: np.ndarray, probability: np.ndarray
    ) -> Dict[str, float]:
        """Calculates Calibration Intercept and Slope using GLM."""
        probability = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
        predicted_logit = np.log(probability / (1 - probability))
        calibration_X = add_constant(predicted_logit, has_constant="add")

        try:
            calibration_model = GLM(
                np.asarray(y_true, dtype=int), calibration_X, family=Binomial()
            ).fit()

            return {
                "Calibration intercept": float(calibration_model.params.iloc[0]),
                "Calibration slope": float(calibration_model.params.iloc[1]),
            }
        except Exception:
            return {"Calibration intercept": np.nan, "Calibration slope": np.nan}

    def evaluate_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_splits: List[Tuple[np.ndarray, np.ndarray]],
        model_key: str = "model",
        model_label: str = "Candidate Model",
    ) -> Dict[str, Any]:
        """
        Evaluates the feature set using exactly the same Stratified K-Fold splits.
        Calculates OOF probabilities, fold metrics, VIF, and coefficient stability.
        """
        X_float = X.astype(float)
        y_int = y.astype(int)

        oof_probability = np.full(len(X), np.nan, dtype=float)
        fold_metrics = []
        coefficient_rows = []

        for fold_number, (train_index, valid_index) in enumerate(cv_splits, start=1):
            X_fold_train = add_constant(X_float.iloc[train_index], has_constant="add")
            X_fold_valid = add_constant(X_float.iloc[valid_index], has_constant="add")

            y_fold_train = y_int.iloc[train_index]
            y_fold_valid = y_int.iloc[valid_index]

            fold_model = GLM(y_fold_train, X_fold_train, family=Binomial()).fit()

            fold_probability = np.asarray(fold_model.predict(X_fold_valid), dtype=float)
            oof_probability[valid_index] = fold_probability

            fold_auc = roc_auc_score(y_fold_valid, fold_probability)

            fold_metrics.append(
                {
                    "Model key": model_key,
                    "Model": model_label,
                    "Fold": fold_number,
                    "AUC": float(fold_auc),
                    "Gini": float(2 * fold_auc - 1),
                    "Brier": float(brier_score_loss(y_fold_valid, fold_probability)),
                    "Log loss": float(log_loss(y_fold_valid, fold_probability)),
                    "KS": self._calculate_ks(y_fold_valid, fold_probability),
                }
            )

            for variable, coefficient in fold_model.params.items():
                if variable == "const":
                    continue
                coefficient_rows.append(
                    {
                        "Model key": model_key,
                        "Model": model_label,
                        "Fold": fold_number,
                        "Variable": variable,
                        "Coefficient": float(coefficient),
                        "Negative coefficient": bool(coefficient < 0),
                    }
                )

        if np.isnan(oof_probability).any():
            raise RuntimeError(f"[ERROR] {model_label}: OOF prediction incomplete.")

        fold_metrics_table = pd.DataFrame(fold_metrics)
        pooled_auc = roc_auc_score(y_int, oof_probability)
        calibration = self._calculate_calibration(y_int, oof_probability)
        vif_table = self._calculate_vif(X_float)

        summary = {
            "Model key": model_key,
            "Model": model_label,
            "Number of features": X.shape[1],
            "OOF AUC": float(pooled_auc),
            "OOF Gini": float(2 * pooled_auc - 1),
            "OOF Brier": float(brier_score_loss(y_int, oof_probability)),
            "OOF Log loss": float(log_loss(y_int, oof_probability)),
            "OOF KS": self._calculate_ks(y_int, oof_probability),
            "Mean CV AUC": float(fold_metrics_table["AUC"].mean()),
            "SD CV AUC": float(fold_metrics_table["AUC"].std(ddof=1)),
            "SE CV AUC": float(
                fold_metrics_table["AUC"].std(ddof=1) / np.sqrt(len(fold_metrics_table))
            ),
            "Minimum fold AUC": float(fold_metrics_table["AUC"].min()),
            "Maximum fold AUC": float(fold_metrics_table["AUC"].max()),
            "Maximum VIF": float(vif_table["VIF"].max()),
            "Any VIF > 5": bool((vif_table["VIF"] > self.vif_threshold).any()),
            **calibration,
        }

        coefficient_table = pd.DataFrame(coefficient_rows)
        coefficient_stability = coefficient_table.groupby(
            ["Model key", "Model", "Variable"], as_index=False
        ).agg(
            Mean_coefficient=("Coefficient", "mean"),
            SD_coefficient=("Coefficient", "std"),
            Negative_fold_rate=("Negative coefficient", "mean"),
        )

        return {
            "summary": summary,
            "oof_probability": oof_probability,
            "fold_metrics": fold_metrics_table,
            "vif": vif_table,
            "fold_coefficients": coefficient_table,
            "coefficient_stability": coefficient_stability,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CreditModelTrainer":
        """
        Fits the final logistic regression model (via statsmodels GLM) on the entire dataset.
        Extracts comprehensive statistical audits (p-values, CI, odds ratios).
        """
        self.final_features_ = list(X.columns)
        self.coefficients_.clear()
        self.audit_report_.clear()

        print("\n" + "=" * 80)
        print("🏛️ [MODEL ENGINE] FITTING FINAL GLM MODEL & EXTRACTING STATISTICS")
        print("=" * 80)

        X_float = X.astype(float)
        y_int = y.astype(int)

        X_const = add_constant(X_float, has_constant="add")
        self.model = GLM(y_int, X_const, family=Binomial()).fit()

        self.intercept_ = float(self.model.params.iloc[0])
        for feature_name, coef_value in self.model.params.items():
            if feature_name == "const":
                self.intercept_ = float(coef_value)
            else:
                self.coefficients_[feature_name] = float(coef_value)

        confidence_interval = self.model.conf_int()

        coefficient_table = pd.DataFrame(
            {
                "Variable": self.model.params.index,
                "Coefficient": self.model.params.values,
                "Standard error": self.model.bse.values,
                "Z statistic": self.model.tvalues.values,
                "P-value": self.model.pvalues.values,
                "CI lower": confidence_interval[0].values,
                "CI upper": confidence_interval[1].values,
            }
        )

        coefficient_table["Odds ratio"] = np.exp(coefficient_table["Coefficient"])
        coefficient_table["Expected WOE sign"] = np.where(
            coefficient_table["Variable"] == "const",
            "Not applicable",
            np.where(coefficient_table["Coefficient"] < 0, "Pass", "Review"),
        )

        self.audit_report_ = {
            "model_summary": str(self.model.summary()),
            "coefficient_statistics": coefficient_table.to_dict(orient="records"),
        }

        print(
            f"  -> [SUCCESS] GLM Model fitted with {len(self.final_features_)} features."
        )
        return self

    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts the raw continuous Probability of Default (PD)."""
        if self.model is None:
            raise ValueError("[ERROR] Model must be fitted before running prediction.")

        X_const = add_constant(
            X[self.final_features_].astype(float), has_constant="add"
        )
        return np.asarray(self.model.predict(X_const), dtype=float)

    def predict_class(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary hard labels based on a probability threshold."""
        probs = self.predict_probability(X)
        return (probs >= threshold).astype(int)
