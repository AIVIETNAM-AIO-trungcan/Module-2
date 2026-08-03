"""
Preprocessing Pipeline for Credit Scoring
-----------------------------------------
This module splits the preprocessing into 2 separate steps:
1. CreditDataCleaner: Standardizes values, handles extreme outliers via Capping,
   enforces strict business logic, and manages missing values dynamically.
   Features a strict 'notebook_match' bypass mode for legacy validation.
2. WOETransformer: Powered by `optbinning` for optimal monotonic binning, WOE encoding,
   and strict mathematical integrity validation (auto-diagnostics).
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

from optbinning import OptimalBinning


# ==============================================================================
# TIER 1: DATA CLEANER (DUAL-MODE)
# ==============================================================================
class CreditDataCleaner:
    """
    Step 1: Cleans the raw credit dataset based on the injected configuration mode.
    Maintains a robust audit registry to log structural mutations for MLOps tracking.
    """

    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        cleaning_config: Dict[str, Any] = None,
        validation_config: Optional[Dict[str, Any]] = None,
        mode: str = "mlops_optimized",
    ) -> None:

        self.numerical_features: List[str] = numerical_features.copy()
        self.categorical_features: List[str] = categorical_features.copy()
        self.cleaning_config: Dict[str, Any] = cleaning_config or {}
        self.validation_config: Dict[str, Any] = validation_config or {}
        self.mode: str = mode
        self.audit_report_: Dict[str, Any] = {}

        # [AUTO-CORRECTION] Smart feature routing based on the operational mode
        if self.mode == "mlops_optimized":
            if "loan_percent_income" in self.numerical_features:
                self.numerical_features.remove("loan_percent_income")
            if "loan_percent_income_computed" not in self.numerical_features:
                self.numerical_features.append("loan_percent_income_computed")

        elif self.mode == "notebook_match":
            if "loan_percent_income_computed" in self.numerical_features:
                self.numerical_features.remove("loan_percent_income_computed")
            if "loan_percent_income" not in self.numerical_features:
                self.numerical_features.append("loan_percent_income")

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "CreditDataCleaner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clean = X.copy()
        total_records = len(X_clean)

        reason_codes: Dict[str, int] = {}
        mutation_audit_log: List[Dict[str, Any]] = []
        dropped_outliers_count = 0
        dropped_outliers_ratio = 0.0
        missing_counts = {}

        # ======================================================================
        # BRANCH 1: NOTEBOOK MATCH (STRICT BYPASS)
        # ======================================================================
        if self.mode == "notebook_match":
            print(
                "  -> [TIER-1 MODE: NOTEBOOK MATCH] Bypassing Data Cleaner to ensure 100% mathematical match with legacy notebook..."
            )
            # We do NOT drop duplicates, do NOT enforce business logic, do NOT upper-case.
            # Data loader has already intercepted the 902 anomaly rows.
            missing_counts = {"Imputation Disabled (Notebook Mode)": 0}

        # ======================================================================
        # BRANCH 2: MLOPS OPTIMIZED (Z-SCORE CAPPING & SAFE IMPUTATION)
        # ======================================================================
        elif self.mode == "mlops_optimized":
            print(
                "  -> [TIER-1 MODE: MLOPS OPTIMIZED] Executing statistical capping and business rules..."
            )

            # 1. Statistical Outlier Handling
            outlier_config = self.cleaning_config.get("outlier_handling", {})
            strategy = outlier_config.get("strategy", "capping")

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
                            mean_val, std_val = series.mean(), series.std()
                            lower_bound = (
                                mean_val - z_thresh * std_val
                                if std_val > 0
                                else mean_val
                            )
                            upper_bound = (
                                mean_val + z_thresh * std_val
                                if std_val > 0
                                else mean_val
                            )
                        elif method == "iqr":
                            q1, q3 = series.quantile(0.25), series.quantile(0.75)
                            iqr_val = q3 - q1
                            lower_bound, upper_bound = (
                                q1 - iqr_mult * iqr_val,
                                q3 + iqr_mult * iqr_val,
                            )

                        if strategy == "drop":
                            outlier_idx = series[
                                (series < lower_bound) | (series > upper_bound)
                            ].index
                            indices_to_drop.update(outlier_idx.tolist())
                        elif strategy == "capping":
                            outlier_mask = (X_clean[col] < lower_bound) | (
                                X_clean[col] > upper_bound
                            )
                            if outlier_mask.sum() > 0:
                                X_clean[col] = np.clip(
                                    X_clean[col], lower_bound, upper_bound
                                )
                                reason_codes[f"CAPPED_OUTLIERS_{col}"] = int(
                                    outlier_mask.sum()
                                )

                if strategy == "drop" and indices_to_drop:
                    dropped_outliers_count = len(indices_to_drop)
                    dropped_outliers_ratio = (
                        dropped_outliers_count / total_records
                    ) * 100
                    reason_codes[f"DROPPED_OUTLIERS_{method.upper()}"] = (
                        dropped_outliers_count
                    )
                    X_clean = X_clean.drop(index=list(indices_to_drop))

            # 2. Dynamic Business Validation Boundaries
            AGE_MIN = self.validation_config.get("person_age", {}).get("min", 18)
            AGE_MAX = self.validation_config.get("person_age", {}).get("max", 100)
            INCOME_MIN = self.validation_config.get("person_income", {}).get("min", 0)
            EMP_LENGTH_MIN = self.validation_config.get("person_emp_length", {}).get(
                "min", 0
            )
            EMP_LENGTH_MAX = self.validation_config.get("person_emp_length", {}).get(
                "max", 100
            )
            LOAN_AMOUNT_MIN = self.validation_config.get("loan_amnt", {}).get("min", 0)
            INTEREST_RATE_MIN = self.validation_config.get("loan_int_rate", {}).get(
                "min", 0.0
            )
            CREDIT_HISTORY_MIN = self.validation_config.get(
                "cb_person_cred_hist_length", {}
            ).get("min", 0)

            for column in self.categorical_features:
                if column in X_clean.columns:
                    X_clean[column] = X_clean[column].apply(
                        lambda x: (
                            str(x).strip().upper()
                            if pd.notna(x) and str(x).strip() != ""
                            else np.nan
                        )
                    )

            # Strict Logic: Emp Length >= Biological Age
            if (
                "person_emp_length" in X_clean.columns
                and "person_age" in X_clean.columns
            ):
                invalid_logic_mask = (
                    (X_clean["person_emp_length"] >= X_clean["person_age"])
                    & X_clean["person_emp_length"].notna()
                    & X_clean["person_age"].notna()
                )
                if invalid_logic_mask.sum() > 0:
                    reason_codes["DROPPED_EMP_GE_AGE"] = int(invalid_logic_mask.sum())
                    X_clean = X_clean.drop(index=X_clean[invalid_logic_mask].index)

            # Map validation violations to NaN for OptBinning
            def apply_boundary_check(col_name, lower, upper, code_name):
                if col_name in X_clean.columns:
                    mask = (
                        ~X_clean[col_name].between(lower, upper)
                        & X_clean[col_name].notna()
                    )
                    if mask.sum() > 0:
                        reason_codes[code_name] = int(mask.sum())
                        for idx in X_clean[mask].index:
                            mutation_audit_log.append(
                                {
                                    "row_index": idx,
                                    "column_altered": col_name,
                                    "original_value": X_clean.loc[idx, col_name],
                                    "imputed_value": "NaN",
                                    "reason": f"Out of valid range [{lower} - {upper}]",
                                }
                            )
                        X_clean.loc[mask, col_name] = np.nan

            apply_boundary_check("person_age", AGE_MIN, AGE_MAX, "INVALID_AGE")
            apply_boundary_check("person_income", INCOME_MIN, np.inf, "INVALID_INCOME")
            apply_boundary_check(
                "person_emp_length",
                EMP_LENGTH_MIN,
                EMP_LENGTH_MAX,
                "INVALID_EMP_LENGTH",
            )
            apply_boundary_check(
                "loan_amnt", LOAN_AMOUNT_MIN, np.inf, "INVALID_LOAN_AMNT"
            )
            apply_boundary_check(
                "loan_int_rate", INTEREST_RATE_MIN, np.inf, "INVALID_INT_RATE"
            )
            apply_boundary_check(
                "cb_person_cred_hist_length",
                CREDIT_HISTORY_MIN,
                np.inf,
                "INVALID_CRED_HIST",
            )

            # Recompute loan percentage
            if {"loan_amnt", "person_income"}.issubset(X_clean.columns):
                X_clean["loan_percent_income_computed"] = (
                    X_clean["loan_amnt"] / X_clean["person_income"].replace(0, np.nan)
                ).round(4)
            X_clean = X_clean.drop(columns=["loan_percent_income"], errors="ignore")

            missing_counts = X_clean.isna().sum().to_dict()

            # Optional Imputation
            imputation_config = self.cleaning_config.get("imputation", {})
            if imputation_config.get("enabled", False) and self.numerical_features:
                num_cols = [c for c in self.numerical_features if c in X_clean.columns]
                X_clean[num_cols] = X_clean[num_cols].fillna(-1.0)

            if self.categorical_features:
                cat_cols = [
                    c for c in self.categorical_features if c in X_clean.columns
                ]
                X_clean[cat_cols] = X_clean[cat_cols].fillna("Missing")

        # ======================================================================
        # MLOPS AUDIT REGISTRY COMMIT
        # ======================================================================
        self.audit_report_ = {
            "total_input_records": total_records,
            "final_records": len(X_clean),
            "outliers_dropped_count": dropped_outliers_count,
            "outliers_dropped_ratio": dropped_outliers_ratio,
            "reason_codes_flagged": reason_codes,
            "missing_values_imputed": missing_counts,
            "mutation_details": mutation_audit_log,
        }

        return X_clean

    def _print_audit_log(self) -> None:
        if not self.audit_report_:
            return
        print(
            f"[DATA QUALITY AUDIT] Total Population Evaluated: {self.audit_report_['total_input_records']} rows"
        )
        if self.mode == "notebook_match":
            print("  -> [NOTEBOOK MODE] OptBinning raw bypass engaged.")
            return

        drop_count = self.audit_report_.get("outliers_dropped_count", 0)
        if drop_count > 0:
            print(
                f"  -> [STATISTICAL] Dropped {drop_count} records to preserve monotonicity."
            )
        capped_count = sum(
            v
            for k, v in self.audit_report_.get("reason_codes_flagged", {}).items()
            if "CAPPED" in k
        )
        if capped_count > 0:
            print(f"  -> [STATISTICAL] Capped {capped_count} extreme outlier values.")
        biz_drop_count = self.audit_report_.get("reason_codes_flagged", {}).get(
            "DROPPED_EMP_GE_AGE", 0
        )
        if biz_drop_count > 0:
            print(
                f"  -> [BUSINESS RULE] Dropped {biz_drop_count} impossible logic records."
            )

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
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

        self.numerical_features = set(numerical_features)
        self.categorical_features = set(categorical_features)
        self.bin_config: Dict[str, Any] = bin_config
        self.small_bin_threshold: float = diagnostics_config.get(
            "small_bin_threshold", 0.05
        )
        self.extreme_woe_threshold: float = diagnostics_config.get(
            "extreme_woe_threshold", 2.0
        )

        self.models_: Dict[str, OptimalBinning] = {}
        self.iv_scores: Dict[str, float] = {}
        self.binning_tables_: Dict[str, pd.DataFrame] = {}
        self.woe_dictionaries: Dict[str, Dict[str, float]] = {}
        self.bin_edges: Dict[str, List[float]] = {}
        self.diagnostic_warnings_: List[Dict[str, Any]] = []

        # [AUTO-CORRECTION] Ensure proper feature routing dynamically based on injected config
        if (
            "loan_percent_income" in self.bin_config
            and "loan_percent_income" not in self.numerical_features
        ):
            self.numerical_features.add("loan_percent_income")
        if (
            "loan_percent_income_computed" in self.bin_config
            and "loan_percent_income_computed" not in self.numerical_features
        ):
            self.numerical_features.add("loan_percent_income_computed")

    def _prepare_feature_array(self, data: pd.DataFrame, feature: str) -> np.ndarray:
        if feature in self.categorical_features:
            return np.array(
                data[feature].fillna("Missing").astype(str).tolist(), dtype=object
            )
        return pd.to_numeric(data[feature], errors="coerce").to_numpy()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOETransformer":
        y_array: np.ndarray = y.astype(int).to_numpy()
        n_train: int = len(X)

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

            # [UPDATED] Dynamically pull max_n_bins from global config (fallback to 6 for Notebook)
            params = {
                "name": feature,
                "dtype": f_type,
                "prebinning_method": self.bin_config.get("prebinning_method", "cart"),
                "solver": self.bin_config.get("solver", "cp"),
                "divergence": self.bin_config.get("divergence", "iv"),
                "max_n_prebins": self.bin_config.get("max_n_prebins", 20),
                "min_prebin_size": self.bin_config.get("min_prebin_size", 0.05),
                "max_n_bins": self.bin_config.get(
                    feature, self.bin_config.get("max_n_bins", 6)
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
                    f"[CRITICAL] OptBinning failed to converge for '{feature}': {e}"
                )

            full_table = model.binning_table.build()
            iv_value = float(model.binning_table.iv)
            status = getattr(model, "status", "UNKNOWN")

            self.models_[feature] = model
            self.iv_scores[feature] = iv_value
            self.binning_tables_[feature] = full_table

            if f_type == "numerical":
                splits = getattr(model, "splits", [])
                self.bin_edges[feature] = [-np.inf] + list(splits) + [np.inf]

            clean_dict_table = full_table.loc[
                ~full_table.index.astype(str).str.strip().isin(["Totals", "Total"])
            ].copy()

            if "Bin" in clean_dict_table.columns:
                clean_dict_table = clean_dict_table.loc[
                    ~clean_dict_table["Bin"]
                    .astype(str)
                    .str.strip()
                    .isin(["Totals", "Total"])
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

            self._run_binning_diagnostics(feature, full_table, n_train)

            print(
                f"  -> {feature:<30} | Type: {f_type:<12} | Status: {status:<10} | Bins: {len(full_table)-2:<3} | IV: {iv_value:.5f}"
            )

        return self

    def _run_binning_diagnostics(
        self, feature: str, table: pd.DataFrame, n_train: int
    ) -> None:
        clean_table = table.loc[
            ~table.index.astype(str).str.strip().isin(["Totals", "Total"])
        ].copy()

        for _, row in clean_table.iterrows():
            if pd.isna(row.get("Count")):
                continue
            count = int(row["Count"])
            if count == 0:
                continue

            share = count / n_train
            woe = float(row["WoE"])
            is_small = share < self.small_bin_threshold
            is_extreme = abs(woe) > self.extreme_woe_threshold
            is_pure = int(row["Non-event"]) == 0 or int(row["Event"]) == 0

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
        X_woe = pd.DataFrame(index=X.index)

        for feature, model in self.models_.items():
            if feature not in X.columns:
                raise KeyError(
                    f"[CRITICAL] Missing required feature for WOE Transform: {feature}"
                )
            x_array = self._prepare_feature_array(X, feature)
            X_woe[feature] = model.transform(
                x_array,
                metric="woe",
                metric_missing="empirical",
                metric_special="empirical",
            )

        self._validate_woe_integrity(X_woe)
        return X_woe

    def transform_to_bins(self, X: pd.DataFrame) -> pd.DataFrame:
        X_bins = pd.DataFrame(index=X.index)

        for feature in self.models_.keys():
            if feature not in X.columns:
                raise KeyError(
                    f"[CRITICAL] Missing required feature for Bin Transform: {feature}"
                )

            if feature in self.categorical_features:
                X_bins[feature] = X[feature].fillna("Missing").astype(str)
            else:
                edges = self.bin_edges[feature]
                group_names = [f"Bin_{i}" for i in range(len(edges) - 1)]
                X_bins[feature] = pd.cut(
                    X[feature], bins=edges, include_lowest=True, labels=group_names
                ).astype(str)
                X_bins[feature] = X_bins[feature].replace("nan", "Missing")

        return X_bins

    def _validate_woe_integrity(self, X_woe: pd.DataFrame) -> None:
        missing_count = int(X_woe.isna().sum().sum())
        infinite_count = int(np.isinf(X_woe.to_numpy(dtype=float)).sum())

        if missing_count > 0:
            raise ValueError(
                f"[DQ-03 FAILED] WOE transformation generated {missing_count} NaN values."
            )
        if infinite_count > 0:
            raise ValueError(
                f"[DQ-03 FAILED] WOE transformation generated {infinite_count} Infinite values."
            )

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
