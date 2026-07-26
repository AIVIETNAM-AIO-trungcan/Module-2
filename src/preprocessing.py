"""
Preprocessing Pipeline for Credit Scoring
-----------------------------------------
This module splits the preprocessing into 2 separate steps:
1. CreditDataCleaner: Standardizes values, handles outliers, and imputes missing values
   dynamically based on validation rules from config.yaml.
2. WOETransformer: Powered by `optbinning` for optimal monotonic binning, WOE encoding,
   and strict mathematical integrity validation (auto-diagnostics).

*Note: Feature Selection (IV screening & LASSO) has been decoupled into a separate
module to prevent data leakage during cross-validation.*
"""

import re
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any

# ==============================================================================
# MONKEY PATCH FOR OPTBINNING & SCIKIT-LEARN 1.5.0+ COMPATIBILITY
# ==============================================================================
import sklearn.utils.validation
import sklearn.utils

# Store the original scikit-learn function
_original_check_array = sklearn.utils.validation.check_array


def _patched_check_array(*args: Any, **kwargs: Any) -> Any:
    """
    Intercepts the scikit-learn check_array call to automatically rename the
    deprecated 'force_all_finite' parameter to 'ensure_all_finite', preventing
    TypeError crashes when optbinning is used with newer scikit-learn versions.
    """
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _original_check_array(*args, **kwargs)


# Override scikit-learn validation functions with the patched version
sklearn.utils.validation.check_array = _patched_check_array
sklearn.utils.check_array = _patched_check_array

# Import optbinning after the patch is applied
from optbinning import OptimalBinning


# ==============================================================================
# TIER 1: DATA CLEANER
# ==============================================================================
class CreditDataCleaner:
    """
    Step 1: Cleans the raw credit dataset by removing invalid records,
    standardizing categorical values, creating derived features,
    and handling missing values dynamically based on config.yaml.
    """

    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        cleaning_config: Dict[str, Any],
    ) -> None:
        """
        Initialize the Data Cleaner.

        Args:
            numerical_features (List[str]): List of numerical feature column names.
            categorical_features (List[str]): List of categorical feature column names.
            cleaning_config (Dict[str, Any]): Dictionary containing min/max boundary validation rules.
        """
        self.numerical_features: List[str] = numerical_features.copy()
        self.categorical_features: List[str] = categorical_features.copy()
        self.cleaning_config: Dict[str, Any] = cleaning_config

        # [AUDIT REGISTRY]: Tracks reason codes without breaking pipeline interface
        self.audit_report_: Dict[str, Any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "CreditDataCleaner":
        """
        Fits the cleaner to the data.
        Nothing is learned mathematically because all cleaning rules are predefined.

        Args:
            X (pd.DataFrame): Training feature matrix.
            y (Optional[pd.Series]): Target values.

        Returns:
            CreditDataCleaner: The fitted instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the dataset using dynamic boundaries and logs out-of-bound mutations.

        Args:
            X (pd.DataFrame): Raw input DataFrame.

        Returns:
            pd.DataFrame: A fully cleaned and imputed DataFrame.
        """
        X_clean = X.copy()
        total_records = len(X_clean)

        # [TEAM HINT - QA DQ-02]: Reason codes dictionary tracking rule violations
        reason_codes: Dict[str, int] = {}
        mutation_audit_log: List[Dict[str, Any]] = []

        # Extract dynamic validation boundaries from config
        AGE_MIN = self.cleaning_config.get("person_age", {}).get("min", 18)
        AGE_MAX = self.cleaning_config.get("person_age", {}).get("max", 100)
        INCOME_MIN = self.cleaning_config.get("person_income", {}).get("min", 0)
        EMP_LENGTH_MIN = self.cleaning_config.get("person_emp_length", {}).get("min", 0)
        EMP_LENGTH_MAX = self.cleaning_config.get("person_emp_length", {}).get(
            "max", 100
        )
        LOAN_AMOUNT_MIN = self.cleaning_config.get("loan_amnt", {}).get("min", 0)
        INTEREST_RATE_MIN = self.cleaning_config.get("loan_int_rate", {}).get(
            "min", 0.0
        )
        CREDIT_HISTORY_MIN = self.cleaning_config.get(
            "cb_person_cred_hist_length", {}
        ).get("min", 0)

        # Standardize categorical text format
        categorical_to_standardize = [
            "person_home_ownership",
            "loan_intent",
            "loan_grade",
            "cb_person_default_on_file",
        ]

        for column in categorical_to_standardize:
            if column in X_clean.columns:
                X_clean[column] = X_clean[column].apply(
                    lambda x: (
                        str(x).strip().upper()
                        if pd.notna(x) and str(x).strip() != ""
                        else np.nan
                    )
                )

        # [ROW PRESERVATION FIX & MUTATION LOGGING]: Assign invalid values to NaN, but log them first
        if "person_age" in X_clean.columns:
            invalid_age_mask = (
                ~X_clean["person_age"].between(AGE_MIN, AGE_MAX)
                & X_clean["person_age"].notna()
            )
            reason_codes["INVALID_AGE"] = int(invalid_age_mask.sum())
            if reason_codes["INVALID_AGE"] > 0:
                for idx in X_clean[invalid_age_mask].index:
                    mutation_audit_log.append(
                        {
                            "row_index": idx,
                            "column_altered": "person_age",
                            "original_value": X_clean.loc[idx, "person_age"],
                            "imputed_value": "NaN",
                            "reason": f"Out of valid range [{AGE_MIN} - {AGE_MAX}]",
                        }
                    )
            X_clean.loc[invalid_age_mask, "person_age"] = np.nan

        if "person_income" in X_clean.columns:
            invalid_income_mask = (X_clean["person_income"] < INCOME_MIN) & X_clean[
                "person_income"
            ].notna()
            reason_codes["INVALID_INCOME"] = int(invalid_income_mask.sum())
            if reason_codes["INVALID_INCOME"] > 0:
                for idx in X_clean[invalid_income_mask].index:
                    mutation_audit_log.append(
                        {
                            "row_index": idx,
                            "column_altered": "person_income",
                            "original_value": X_clean.loc[idx, "person_income"],
                            "imputed_value": "NaN",
                            "reason": f"Below minimum (< {INCOME_MIN})",
                        }
                    )
            X_clean.loc[invalid_income_mask, "person_income"] = np.nan

        if "person_emp_length" in X_clean.columns:
            invalid_emp_mask = (
                ~X_clean["person_emp_length"].between(EMP_LENGTH_MIN, EMP_LENGTH_MAX)
                & X_clean["person_emp_length"].notna()
            )
            reason_codes["INVALID_EMP_LENGTH"] = int(invalid_emp_mask.sum())
            if reason_codes["INVALID_EMP_LENGTH"] > 0:
                for idx in X_clean[invalid_emp_mask].index:
                    mutation_audit_log.append(
                        {
                            "row_index": idx,
                            "column_altered": "person_emp_length",
                            "original_value": X_clean.loc[idx, "person_emp_length"],
                            "imputed_value": "NaN",
                            "reason": f"Out of valid range [{EMP_LENGTH_MIN} - {EMP_LENGTH_MAX}]",
                        }
                    )
            X_clean.loc[invalid_emp_mask, "person_emp_length"] = np.nan

        if "loan_amnt" in X_clean.columns:
            invalid_loan_mask = (X_clean["loan_amnt"] < LOAN_AMOUNT_MIN) & X_clean[
                "loan_amnt"
            ].notna()
            reason_codes["INVALID_LOAN_AMNT"] = int(invalid_loan_mask.sum())
            if reason_codes["INVALID_LOAN_AMNT"] > 0:
                for idx in X_clean[invalid_loan_mask].index:
                    mutation_audit_log.append(
                        {
                            "row_index": idx,
                            "column_altered": "loan_amnt",
                            "original_value": X_clean.loc[idx, "loan_amnt"],
                            "imputed_value": "NaN",
                            "reason": f"Below minimum (< {LOAN_AMOUNT_MIN})",
                        }
                    )
            X_clean.loc[invalid_loan_mask, "loan_amnt"] = np.nan

        if "loan_int_rate" in X_clean.columns:
            invalid_rate_mask = (
                X_clean["loan_int_rate"] < INTEREST_RATE_MIN
            ) & X_clean["loan_int_rate"].notna()
            reason_codes["INVALID_INT_RATE"] = int(invalid_rate_mask.sum())
            if reason_codes["INVALID_INT_RATE"] > 0:
                for idx in X_clean[invalid_rate_mask].index:
                    mutation_audit_log.append(
                        {
                            "row_index": idx,
                            "column_altered": "loan_int_rate",
                            "original_value": X_clean.loc[idx, "loan_int_rate"],
                            "imputed_value": "NaN",
                            "reason": f"Below minimum (< {INTEREST_RATE_MIN})",
                        }
                    )
            X_clean.loc[invalid_rate_mask, "loan_int_rate"] = np.nan

        if "cb_person_cred_hist_length" in X_clean.columns:
            invalid_hist_mask = (
                X_clean["cb_person_cred_hist_length"] < CREDIT_HISTORY_MIN
            ) & X_clean["cb_person_cred_hist_length"].notna()
            reason_codes["INVALID_CRED_HIST"] = int(invalid_hist_mask.sum())
            if reason_codes["INVALID_CRED_HIST"] > 0:
                for idx in X_clean[invalid_hist_mask].index:
                    mutation_audit_log.append(
                        {
                            "row_index": idx,
                            "column_altered": "cb_person_cred_hist_length",
                            "original_value": X_clean.loc[
                                idx, "cb_person_cred_hist_length"
                            ],
                            "imputed_value": "NaN",
                            "reason": f"Below minimum (< {CREDIT_HISTORY_MIN})",
                        }
                    )
            X_clean.loc[invalid_hist_mask, "cb_person_cred_hist_length"] = np.nan

        # Recompute loan percentage of income to fix data inconsistencies
        if {"loan_amnt", "person_income"}.issubset(X_clean.columns):
            X_clean["loan_percent_income_computed"] = (
                X_clean["loan_amnt"] / X_clean["person_income"].replace(0, np.nan)
            ).round(4)

        # Remove original loan percentage feature to prevent multicollinearity
        X_clean = X_clean.drop(columns=["loan_percent_income"], errors="ignore")

        # Track missing value counts before imputation
        missing_counts = X_clean.isna().sum().to_dict()

        # Fill missing numerical values with constant -1.0 flag
        if self.numerical_features:
            numerical_columns = [
                col for col in self.numerical_features if col in X_clean.columns
            ]
            X_clean[numerical_columns] = X_clean[numerical_columns].fillna(-1.0)

        # Fill missing categorical values with string "Missing" flag
        if self.categorical_features:
            categorical_columns = [
                col for col in self.categorical_features if col in X_clean.columns
            ]
            X_clean[categorical_columns] = X_clean[categorical_columns].fillna(
                "Missing"
            )

        # Save audit registry
        self.audit_report_ = {
            "total_input_records": total_records,
            "reason_codes_flagged": reason_codes,
            "missing_values_imputed": missing_counts,
            "mutation_details": mutation_audit_log,
        }

        return X_clean

    def _print_audit_log(self) -> None:
        """Helper method to print the data quality audit report to the terminal."""
        if not self.audit_report_:
            print(
                "[DATA QUALITY AUDIT] No audit report generated yet. Run transform() first."
            )
            return

        print(
            f"[DATA QUALITY AUDIT] Total Population Evaluated: {self.audit_report_['total_input_records']} rows"
        )

        flags = self.audit_report_.get("reason_codes_flagged", {})
        if any(v > 0 for v in flags.values()):
            print("  -> Flagged Invalid Outliers (Converted to NaN):")
            for code, count in flags.items():
                if count > 0:
                    print(f"     * [{code}]: {count} records")

        mutation_log = self.audit_report_.get("mutation_details", [])
        if mutation_log:
            print(
                f"  -> [MUTATION AUDIT] Captured {len(mutation_log)} mutation events. Tracked in audit_report_['mutation_details']."
            )

        imputed = self.audit_report_.get("missing_values_imputed", {})
        if imputed:
            print("  -> Imputed Missing Values:")
            for col, count in imputed.items():
                if count > 0:
                    print(f"     * [{col}]: {count} rows imputed")

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Combines fit() and transform() sequentially.
        """
        return self.fit(X, y).transform(X)


# ==============================================================================
# TIER 2: OPTIMAL WOE TRANSFORMER
# ==============================================================================
class WOETransformer:
    """
    Step 2: Uses `optbinning` for optimal risk-based discretization.
    Includes built-in validation for WOE mapping integrity and bin diagnostics.
    """

    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        bin_config: Dict[str, Any],
        diagnostics_config: Dict[str, float],
    ) -> None:
        """
        Initializes the Weight of Evidence (WOE) transformer.

        Args:
            numerical_features (List[str]): Numerical columns to be discretized.
            categorical_features (List[str]): Categorical columns to be encoded.
            bin_config (Dict[str, Any]): Dictionary of settings for optbinning.
            diagnostics_config (Dict[str, float]): Thresholds for bin auditing.
        """
        self.numerical_features = set(numerical_features)
        self.categorical_features = set(categorical_features)

        # Configuration mapping
        self.bin_config: Dict[str, Any] = bin_config
        self.small_bin_threshold: float = diagnostics_config.get(
            "small_bin_threshold", 0.05
        )
        self.extreme_woe_threshold: float = diagnostics_config.get(
            "extreme_woe_threshold", 2.0
        )

        # Stores the OptBinning models, IV scores, and tables
        self.models_: Dict[str, OptimalBinning] = {}
        self.iv_scores: Dict[str, float] = {}
        self.binning_tables_: Dict[str, pd.DataFrame] = {}

        # Backward compatibility for Step 7 scorecard generation
        self.woe_dictionaries: Dict[str, Dict[str, float]] = {}
        self.bin_edges: Dict[str, List[float]] = {}

        # MLOps Diagnostics Registry (Tracks small bins, pure bins, extreme WOE)
        self.diagnostic_warnings_: List[Dict[str, Any]] = []

    def _prepare_feature_array(self, data: pd.DataFrame, feature: str) -> np.ndarray:
        """
        Prepares column dtype correctly for OptBinning solver input.
        Strips PyArrow string wrappers to ensure clean labels in downstream charts.
        """
        if feature in self.categorical_features:
            return np.array(data[feature].astype(str).tolist(), dtype=object)
        return pd.to_numeric(data[feature], errors="coerce").to_numpy()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOETransformer":
        """
        Learns the optimal bin boundaries and calculates the explicit WOE math.

        Args:
            X (pd.DataFrame): Training data (must be pre-cleaned by Tier 1).
            y (pd.Series): Target labels (1 for Default/Event, 0 for Good/Non-Event).

        Returns:
            WOETransformer: The fitted instance.
        """
        y_array: np.ndarray = y.astype(int).to_numpy()
        n_train: int = len(X)

        # Safely intersect defined features with actual columns available
        all_features: List[str] = list(
            self.numerical_features.union(self.categorical_features).intersection(
                X.columns
            )
        )

        print("\n" + "=" * 80)
        print("⚙️ [WOE ENGINE] INITIATING OPTIMAL BINNING MATHEMATICS")
        print("=" * 80)

        for feature in all_features:
            f_type = (
                "categorical" if feature in self.categorical_features else "numerical"
            )

            # Setup Special Codes matching Tier 1 imputations
            special_codes = ["Missing"] if f_type == "categorical" else [-1.0]

            # Dynamic parameters overriding mechanism
            params = {
                "name": feature,
                "dtype": f_type,
                "prebinning_method": self.bin_config.get("prebinning_method", "cart"),
                "solver": self.bin_config.get("solver", "cp"),
                "divergence": self.bin_config.get("divergence", "iv"),
                "max_n_prebins": self.bin_config.get("max_n_prebins", 20),
                "min_prebin_size": self.bin_config.get("min_prebin_size", 0.05),
                "max_n_bins": self.bin_config.get(
                    feature, self.bin_config.get("default_bins", 6)
                ),
                "special_codes": special_codes,
            }

            if f_type == "numerical":
                params["monotonic_trend"] = self.bin_config.get(
                    "monotonic_trend", "auto"
                )

            model = OptimalBinning(**params)
            x_array = self._prepare_feature_array(X, feature)

            try:
                model.fit(x_array, y_array)
            except Exception as e:
                raise RuntimeError(
                    f"[CRITICAL] OptBinning failed to converge for feature '{feature}': {e}"
                )

            # Extract Table & Metrics
            full_table = model.binning_table.build()
            iv_value = float(model.binning_table.iv)
            status = getattr(model, "status", "UNKNOWN")

            self.models_[feature] = model
            self.iv_scores[feature] = iv_value
            self.binning_tables_[feature] = full_table

            # Record numerical splits for scoring pipelines
            if f_type == "numerical":
                splits = getattr(model, "splits", [])
                # Insert -inf and inf as bounds to be utilized by pd.cut() downstream
                self.bin_edges[feature] = [-np.inf] + list(splits) + [np.inf]

            # Reconstruct woe_dictionaries for backward compatibility (used in Scorecard construction)
            clean_dict_table = full_table.loc[
                ~full_table.index.astype(str).str.strip().isin(["Totals", "Total"])
            ]

            if f_type == "categorical":
                feature_woe_dict = {}
                for _, row in clean_dict_table.iterrows():
                    bin_val = row["Bin"]
                    woe_val = float(row["WoE"])
                    bin_str = str(bin_val)

                    if isinstance(bin_val, (list, np.ndarray)):
                        for cat in bin_val:
                            feature_woe_dict[str(cat).strip()] = woe_val
                    elif (
                        "[" in bin_str
                        and "]" in bin_str
                        and ("'" in bin_str or '"' in bin_str)
                    ):
                        cats = re.findall(r"['\"](.*?)['\"]", bin_str)
                        for cat in cats:
                            feature_woe_dict[cat.strip()] = woe_val
                    else:
                        feature_woe_dict[bin_str.strip()] = woe_val

                self.woe_dictionaries[feature] = feature_woe_dict
            else:
                self.woe_dictionaries[feature] = dict(
                    zip(clean_dict_table["Bin"].astype(str), clean_dict_table["WoE"])
                )

            # Run Background Analytics/Diagnostics
            self._run_binning_diagnostics(feature, full_table, n_train)

            print(
                f"  -> {feature:<30} | Type: {f_type:<12} | Status: {status:<10} | Bins: {len(full_table)-2:<3} | IV: {iv_value:.5f}"
            )

        return self

    def _run_binning_diagnostics(
        self, feature: str, table: pd.DataFrame, n_train: int
    ) -> None:
        """
        Internal audit for small bins, pure bins, and extreme WOE values.
        Alerts are safely stored in `diagnostic_warnings_` to be exported by MLOps.
        """
        # Filter out "Totals" row
        clean_table = table.loc[
            ~table.index.astype(str).str.strip().isin(["Totals", "Total"])
        ].copy()

        for _, row in clean_table.iterrows():
            if pd.isna(row.get("Count")):
                continue

            count = int(row["Count"])
            if count == 0:
                continue

            good = int(row["Non-event"])
            bad = int(row["Event"])
            share = count / n_train
            woe = float(row["WoE"])

            is_small = share < self.small_bin_threshold
            is_extreme = abs(woe) > self.extreme_woe_threshold
            is_pure = good == 0 or bad == 0

            if is_small or is_extreme or is_pure:
                self.diagnostic_warnings_.append(
                    {
                        "Variable": feature,
                        "Bin": str(row["Bin"]),
                        "Count": count,
                        "Share": f"{share:.2%}",
                        "WOE": round(woe, 4),
                        "Warning_Small_Bin": is_small,
                        "Warning_Extreme_WOE": is_extreme,
                        "Warning_Pure_Bin": is_pure,
                    }
                )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms dataset into WOE scores and rigorously validates integrity.

        Args:
            X (pd.DataFrame): The unseen raw data to transform.

        Returns:
            pd.DataFrame: A dataset where original values are replaced by continuous WOE floats.

        Raises:
            KeyError: If a required feature column is missing.
        """
        X_woe = pd.DataFrame(index=X.index)

        for feature, model in self.models_.items():
            if feature not in X.columns:
                raise KeyError(
                    f"[CRITICAL] Missing required feature for WOE Transform: {feature}"
                )

            x_array = self._prepare_feature_array(X, feature)

            # Use empirical metrics to safeguard against unseen inference data anomalies
            X_woe[feature] = model.transform(
                x_array,
                metric="woe",
                metric_missing="empirical",
                metric_special="empirical",
            )

        # Perform rigid validation before releasing transformed data
        self._validate_woe_integrity(X_woe)

        return X_woe

    def _validate_woe_integrity(self, X_woe: pd.DataFrame) -> None:
        """
        Hard constraint validator: Stops the pipeline if WOE transform produces NaN or Inf.

        Raises:
            ValueError: If missing/infinite values are detected.
        """
        missing_count = int(X_woe.isna().sum().sum())
        infinite_count = int(np.isinf(X_woe.to_numpy(dtype=float)).sum())

        if missing_count > 0:
            raise ValueError(
                f"[DQ-03 FAILED] WOE transformation generated {missing_count} NaN values. Check unseen categories."
            )

        if infinite_count > 0:
            raise ValueError(
                f"[DQ-03 FAILED] WOE transformation generated {infinite_count} Infinite (Inf) values. Check pure bins."
            )

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Combines fit() and transform() sequentially.
        """
        self.fit(X, y)
        return self.transform(X)


# ==============================================================================
# INTERNAL TEST BLOCK
# ==============================================================================
if __name__ == "__main__":
    print("--- 🚀 STARTING COMPREHENSIVE PIPELINE TEST 🚀 ---")

    NUM_COLS = ["person_age", "person_income", "loan_amnt", "person_emp_length"]
    CAT_COLS = [
        "person_home_ownership",
        "loan_intent",
        "noise_feature",
        "strong_cat_feature",
    ]

    # Mocking Config structure
    MOCK_CLEANING_CONFIG = {
        "person_age": {"min": 18, "max": 99},
        "person_income": {"min": 1},
        "person_emp_length": {"min": 0, "max": 99},
        "loan_amnt": {"min": 1},
    }

    MOCK_BIN_CONFIG = {
        "default_bins": 3,
        "person_age": 4,
    }

    MOCK_DIAGNOSTICS_CONFIG = {
        "small_bin_threshold": 0.05,
        "extreme_woe_threshold": 2.0,
    }

    # MOCK DATA
    sample_X = pd.DataFrame(
        {
            "person_age": [
                25,
                30,
                45,
                12,
                150,
                28,
                35,
                40,
                22,
                50,
                29,
                31,
                33,
                27,
                41,
                38,
                24,
                26,
                60,
                -5,
            ],
            "person_income": [
                50000,
                60000,
                120000,
                np.nan,
                85000,
                45000,
                70000,
                np.nan,
                30000,
                200000,
                55000,
                62000,
                80000,
                48000,
                95000,
                72000,
                40000,
                52000,
                110000,
                np.nan,
            ],
            "loan_amnt": [
                10000,
                15000,
                25000,
                5000,
                10000,
                8000,
                12000,
                15000,
                5000,
                35000,
                11000,
                13000,
                20000,
                9000,
                22000,
                14000,
                7000,
                10000,
                30000,
                5000,
            ],
            "loan_percent_income": [0.2] * 20,
            "person_emp_length": [
                10,
                15,
                25,
                5,
                10,
                8,
                12,
                15,
                5,
                35,
                11,
                13,
                20,
                9,
                22,
                14,
                7,
                10,
                30,
                5,
            ],
            "person_home_ownership": [
                "RENT",
                "mortgage",
                "OWN",
                np.nan,
                "OWN ",
                "RENT",
                "RENT",
                "MORTGAGE",
                "RENT",
                "OWN",
                "RENT",
                "MORTGAGE",
                "OWN",
                "RENT",
                "MORTGAGE",
                "RENT",
                "RENT",
                "MORTGAGE",
                "OWN",
                np.nan,
            ],
            "loan_intent": [
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "PERSONAL",
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "PERSONAL",
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "PERSONAL",
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "PERSONAL",
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "PERSONAL",
            ],
            "noise_feature": ["A"] * 20,
            "strong_cat_feature": [
                "X",
                "X",
                "Y",
                "Y",
                "X",
                "X",
                "X",
                "Y",
                "Y",
                "X",
                "X",
                "X",
                "Y",
                "X",
                "Y",
                "X",
                "Y",
                "X",
                "Y",
                "Y",
            ],
        }
    )

    sample_y = pd.Series([0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1])

    # ------------------
    # RUN TIER 1: CLEANING
    # ------------------
    cleaner = CreditDataCleaner(
        numerical_features=NUM_COLS,
        categorical_features=CAT_COLS,
        cleaning_config=MOCK_CLEANING_CONFIG,
    )
    clean_data = cleaner.fit_transform(sample_X)

    # ------------------
    # RUN TIER 2: WOE
    # ------------------
    woe_transformer = WOETransformer(
        numerical_features=cleaner.numerical_features,
        categorical_features=cleaner.categorical_features,
        bin_config=MOCK_BIN_CONFIG,
        diagnostics_config=MOCK_DIAGNOSTICS_CONFIG,
    )
    final_data = woe_transformer.fit_transform(clean_data, sample_y)

    # ------------------
    # SYSTEM AUDIT LOGS
    # ------------------
    print("\n" + "=" * 50)
    print(" 🛠️ [AUDIT] TIER 1: DATA CLEANER")
    print("=" * 50)
    cleaner._print_audit_log()
    print(
        f"  -> [Check]: Dropped old 'loan_percent_income': {'loan_percent_income' not in clean_data.columns}"
    )
    print(
        f"  -> [Check]: Created 'loan_percent_income_computed': {'loan_percent_income_computed' in clean_data.columns}"
    )

    print("\n" + "=" * 50)
    print(" 📊 [AUDIT] BINNING WARNINGS")
    print("=" * 50)
    if woe_transformer.diagnostic_warnings_:
        df_warn = pd.DataFrame(woe_transformer.diagnostic_warnings_)
        print(df_warn.to_string(index=False))
    else:
        print("No Binning Warnings Detected.")

    print("\n==================================================")
    print("[SUCCESS] Preprocessing Pipeline executed flawlessly! 🎉")
