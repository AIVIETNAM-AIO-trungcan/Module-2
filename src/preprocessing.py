"""
Preprocessing Pipeline for Credit Scoring
-----------------------------------------
This module splits the preprocessing into 2 separate steps:
1. CreditDataCleaner: Standardizes values, handles extreme outliers via Capping,
   enforces strict business logic (e.g., Emp Length < Age), and manages missing
   values dynamically based on config.yaml.
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
    Step 1: Cleans the raw credit dataset by capping extreme outliers mathematically,
    removing invalid business records (e.g., employment length >= age),
    standardizing categorical values, and respecting missing values dynamically.
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
            cleaning_config (Dict[str, Any]): Dictionary containing outlier config and boundary validation rules.
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
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the dataset using statistical outlier capping and dynamic business boundaries.

        Args:
            X (pd.DataFrame): Raw input DataFrame.

        Returns:
            pd.DataFrame: A fully cleaned DataFrame.
            The index might be shorter if strictly invalid business logic outliers were dropped.
        """
        X_clean = X.copy()
        total_records = len(X_clean)

        # [TEAM HINT - QA DQ-02]: Reason codes dictionary tracking rule violations
        reason_codes: Dict[str, int] = {}
        mutation_audit_log: List[Dict[str, Any]] = []

        # ======================================================================
        # 1. STATISTICAL OUTLIER HANDLING (Z-SCORE / IQR CAPPING OR DROPPING)
        # ======================================================================
        outlier_config = self.cleaning_config.get("outlier_handling", {})
        strategy = outlier_config.get("strategy", "capping")
        dropped_outliers_count = 0
        dropped_outliers_ratio = 0.0

        if outlier_config.get("enabled", False):
            method = outlier_config.get("method", "zscore")
            target_cols = outlier_config.get("target_columns", [])
            z_thresh = outlier_config.get("zscore_threshold", 3.0)
            iqr_mult = outlier_config.get("iqr_multiplier", 3.0)

            indices_to_drop = set()

            for col in target_cols:
                if col in X_clean.columns:
                    series = X_clean[col].dropna()

                    if method == "zscore":
                        mean_val = series.mean()
                        std_val = series.std()
                        if std_val > 0:
                            lower_bound = mean_val - z_thresh * std_val
                            upper_bound = mean_val + z_thresh * std_val
                        else:
                            lower_bound, upper_bound = mean_val, mean_val

                    elif method == "iqr":
                        q1 = series.quantile(0.25)
                        q3 = series.quantile(0.75)
                        iqr_val = q3 - q1
                        lower_bound = q1 - iqr_mult * iqr_val
                        upper_bound = q3 + iqr_mult * iqr_val

                    if strategy == "drop":
                        outlier_idx = series[
                            (series < lower_bound) | (series > upper_bound)
                        ].index
                        indices_to_drop.update(outlier_idx.tolist())

                    elif strategy == "capping":
                        outlier_mask = (X_clean[col] < lower_bound) | (
                            X_clean[col] > upper_bound
                        )
                        outliers_count = outlier_mask.sum()
                        if outliers_count > 0:
                            X_clean[col] = np.clip(
                                X_clean[col], lower_bound, upper_bound
                            )
                            reason_codes[f"CAPPED_OUTLIERS_{col}"] = int(outliers_count)

            if strategy == "drop" and indices_to_drop:
                dropped_outliers_count = len(indices_to_drop)
                dropped_outliers_ratio = (dropped_outliers_count / total_records) * 100
                reason_codes[f"DROPPED_OUTLIERS_{method.upper()}"] = (
                    dropped_outliers_count
                )
                X_clean = X_clean.drop(index=list(indices_to_drop))

        # ======================================================================
        # 2. BUSINESS LOGICAL BOUNDARY VALIDATION
        # ======================================================================
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

        # [STRICT BUSINESS RULE]: Drop records where Employment Length >= Biological Age
        if "person_emp_length" in X_clean.columns and "person_age" in X_clean.columns:
            invalid_logic_mask = (
                (X_clean["person_emp_length"] >= X_clean["person_age"])
                & X_clean["person_emp_length"].notna()
                & X_clean["person_age"].notna()
            )
            invalid_logic_count = invalid_logic_mask.sum()
            if invalid_logic_count > 0:
                reason_codes["DROPPED_EMP_GE_AGE"] = int(invalid_logic_count)
                X_clean = X_clean.drop(index=X_clean[invalid_logic_mask].index)

        # [MUTATION LOGGING]: Assign invalid single-field boundaries to NaN
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

        # Track missing value counts before imputation (if any)
        missing_counts = X_clean.isna().sum().to_dict()

        # ======================================================================
        # 3. IMPUTATION HANDLING
        # ======================================================================
        imputation_config = self.cleaning_config.get("imputation", {})

        # Tech Lead Rule: Do not impute numerical missing values with Mean/Median.
        # Leave them as NaN so OptBinning assigns them to an isolated 'Missing' Bin for proper risk profiling.
        if imputation_config.get("enabled", False):
            if self.numerical_features:
                numerical_columns = [
                    col for col in self.numerical_features if col in X_clean.columns
                ]
                X_clean[numerical_columns] = X_clean[numerical_columns].fillna(-1.0)

        # For categoricals, we explicitly map NaNs to the string "Missing" to ensure type consistency
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
            "final_records": len(X_clean),
            "outliers_dropped_count": dropped_outliers_count,
            "outliers_dropped_ratio": dropped_outliers_ratio,
            "reason_codes_flagged": reason_codes,
            "missing_values_imputed": (
                missing_counts
                if imputation_config.get("enabled", False)
                else {"Imputation Disabled": 0}
            ),
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

        # Log dropped/capped outliers rationale
        drop_count = self.audit_report_.get("outliers_dropped_count", 0)
        drop_ratio = self.audit_report_.get("outliers_dropped_ratio", 0.0)
        if drop_count > 0:
            print(
                f"  -> [STATISTICAL CLEANING] Permanently dropped {drop_count} extreme outlier records "
                f"({drop_ratio:.4f}% of total data) to preserve risk monotonicity."
            )

        capped_count = sum(
            v
            for k, v in self.audit_report_.get("reason_codes_flagged", {}).items()
            if "CAPPED" in k
        )
        if capped_count > 0:
            print(
                f"  -> [STATISTICAL CLEANING] Successfully capped {capped_count} extreme outlier values to upper/lower boundaries."
            )

        biz_drop_count = self.audit_report_.get("reason_codes_flagged", {}).get(
            "DROPPED_EMP_GE_AGE", 0
        )
        if biz_drop_count > 0:
            print(
                f"  -> [BUSINESS RULE] Permanently dropped {biz_drop_count} records due to impossible logic (Emp Length >= Age)."
            )

        flags = self.audit_report_.get("reason_codes_flagged", {})
        if any(v > 0 for v in flags.values()):
            print("  -> Flagged Anomalies & Violations:")
            for code, count in flags.items():
                if count > 0:
                    action = (
                        "Permanently Dropped"
                        if "DROPPED" in code
                        else (
                            "Capped to Bounds"
                            if "CAPPED" in code
                            else "Converted to NaN"
                        )
                    )
                    print(f"     * [{code}]: {count} records ({action})")

        mutation_log = self.audit_report_.get("mutation_details", [])
        if mutation_log:
            print(
                f"  -> [MUTATION AUDIT] Captured {len(mutation_log)} mutation events. Tracked in audit_report_['mutation_details']."
            )

        imputed = self.audit_report_.get("missing_values_imputed", {})
        if imputed and "Imputation Disabled" not in imputed:
            print("  -> Imputed Missing Values:")
            for col, count in imputed.items():
                if count > 0:
                    print(f"     * [{col}]: {count} rows imputed")
        elif "Imputation Disabled" in imputed:
            print(
                "  -> [MISSING VALUES]: Imputation is DISABLED. NaN values preserved for Risk Binning."
            )

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
            return np.array(
                data[feature].fillna("Missing").astype(str).tolist(), dtype=object
            )
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
                    feature, self.bin_config.get("default_bins", 5)
                ),
                "special_codes": ["Missing"] if f_type == "categorical" else None,
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
                feature_woe_dict = {}
                bin_counter = 0
                for _, row in clean_dict_table.iterrows():
                    bin_original_name = str(row["Bin"]).strip()
                    woe_val = float(row["WoE"])

                    if bin_original_name in ["Special", "Missing"]:
                        feature_woe_dict[bin_original_name] = woe_val
                    else:
                        feature_woe_dict[f"Bin_{bin_counter}"] = woe_val
                        bin_counter += 1

                self.woe_dictionaries[feature] = feature_woe_dict

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
