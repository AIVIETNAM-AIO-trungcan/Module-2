"""
Preprocessing Pipeline for Credit Scoring
-----------------------------------------
This module splits the preprocessing into 2 seperate steps to support different models():
1. CreditDataCleaner: Fill missing values. Used for all models
2. CreditWoETranformer: Converts data to Weight of Evidence (WoE) scores. Used mainly for logistic regression
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any

from src.config import (
    AGE_MIN,
    AGE_MAX,
    INCOME_MIN,
    EMP_LENGTH_MIN,
    EMP_LENGTH_MAX,
    LOAN_AMOUNT_MIN,
    INTEREST_RATE_MIN,
    CREDIT_HISTORY_MIN,
)


# TIER 1: DATA CLEANER
class CreditDataCleaner:
    """
    Step 1: Cleans the raw credit dataset by removing invalid records,
    standardizing categorical values, creating derived features,
    and handling missing values.
    """

    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
    ) -> None:
        """
        Initialize the cleaner.

        Args:
            numerical_features (List[str]):
                List of numerical feature names.

            categorical_features (List[str]):
                List of categorical feature names.
        """
        self.numerical_features = numerical_features.copy()
        self.categorical_features = categorical_features.copy()

        # [AUDIT REGISTRY]: Tracks reason codes without breaking pipeline interface
        self.audit_report_: Dict[str, Any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "CreditDataCleaner":
        """
        Nothing is learned because all cleaning rules are predefined.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset.

        Args:
            X (pd.DataFrame):
                Raw input dataframe.

        Returns:
            pd.DataFrame:
                Cleaned dataframe.
        """
        X_clean = X.copy()
        total_records = len(X_clean)

        # [TEAM HINT - QA DQ-02]: Reason codes dictionary tracking rule violations
        reason_codes: Dict[str, int] = {}

        # Initialize the Audit Log list to track altered data rows
        mutation_audit_log: List[Dict[str, Any]] = []

        # Standardize categorical values
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

        # Recompute loan percentage of income
        if {"loan_amnt", "person_income"}.issubset(X_clean.columns):
            X_clean["loan_percent_income_computed"] = (
                X_clean["loan_amnt"] / X_clean["person_income"].replace(0, np.nan)
            ).round(4)

        # Remove original loan percentage feature
        X_clean = X_clean.drop(columns=["loan_percent_income"], errors="ignore")

        # Track missing value counts before imputation
        missing_counts = X_clean.isna().sum().to_dict()

        # Fill missing numerical values
        if self.numerical_features:
            numerical_columns = [
                col for col in self.numerical_features if col in X_clean.columns
            ]
            X_clean[numerical_columns] = X_clean[numerical_columns].fillna(-1.0)

        # Fill missing categorical values
        if self.categorical_features:
            categorical_columns = [
                col for col in self.categorical_features if col in X_clean.columns
            ]
            X_clean[categorical_columns] = X_clean[categorical_columns].fillna(
                "Missing"
            )

        self.audit_report_ = {
            "total_input_records": total_records,
            "reason_codes_flagged": reason_codes,
            "missing_values_imputed": missing_counts,
            "mutation_details": mutation_audit_log,
        }

        return X_clean

    def _print_audit_log(self) -> None:
        """Helper method to print audit report to terminal."""
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
                print(f"     * [{col}]: {count} rows imputed")

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Fit and transform the dataset.

        Args:
            X (pd.DataFrame):
                Training data.

            y (Optional[pd.Series]):
                Target labels.

        Returns:
            pd.DataFrame:
                Cleaned dataframe.
        """
        return self.fit(X, y).transform(X)


# TIER 2: WOE TRANSFORMER
class WOETransformer:
    """
    Step 2: Group numbers into bins and calculates Weight of Evidence(WoE)
    """

    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        bin_config: Dict[str, int],
    ) -> None:
        """
        Initializes the WOE transformer.

        Args:
            numerical_features (List[str]): Numerical columns to be binned.
            categorical_features (List[str]): Text columns to be encoded.
            n_bins Dict[str, int]: The number of intervals to split numerical data into. Gets from config.yaml
        """
        self.numerical_features: List[str] = numerical_features.copy()
        self.categorical_features: List[str] = categorical_features.copy()
        self.n_bins: Dict[str, int] = bin_config

        # Stores the cut-off points for each numerical column
        self.bin_edges: Dict[str, np.array] = {}

        # Stores the mapping dictionary  from text/bin to WOE score for each column
        self.woe_dictionaries: Dict[str, Dict[str, float]] = {}

        # Stores the IV:
        self.iv_scores: Dict[str, float] = {}
        self.selected_features: List[str] = (
            []
        )  # Add this variable to store features that pass the IV threshold

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOETransformer":
        """
        Learns the bin boundaries and calculates the explicit WOE math from the training data.

        Args:
            X (pd.DataFrame): Training data (must be pre-cleaned by Tier 1).
            y (pd.Series): Target labels (1 for Bad customer, 0 for Good customer).

        Returns:
            TransparentWOETransformer: The fitted instance containing learned dictionaries.
        """
        X_copy: pd.DataFrame = X.copy()

        # Sync with cleaner dynamically added features if present
        for col in X.columns:
            if (
                col not in self.numerical_features
                and col not in self.categorical_features
            ):
                if pd.api.types.is_numeric_dtype(X[col]):
                    self.numerical_features.append(col)

        # 1. FIND BIN EDGES FOR FOR NUMERICAL FEATURES
        for col in self.numerical_features:
            # Exclude the -1.0 missing flag to find true statistical quartiles
            valid_data: pd.Series = X_copy[X_copy[col] != -1.0][col]
            n_bins_for_col = self.n_bins.get(col, self.n_bins.get("default_bins", 5))

            if valid_data.empty or len(valid_data.unique()) <= 1:
                self.bin_edges[col] = np.array([-np.inf, np.inf])
            else:
                _, edges = pd.qcut(
                    valid_data, q=n_bins_for_col, retbins=True, duplicates="drop"
                )
                self.bin_edges[col] = edges

            # FIX LOGIC: Do not merge -1.0 into Bin_0. Keep them separate.
            group_names = [f"Bin_{i}" for i in range(len(self.bin_edges[col]) - 1)]
            binned_col = pd.cut(
                X_copy[col],
                bins=self.bin_edges[col],
                include_lowest=True,
                labels=group_names,
            ).astype(str)

            # Manually isolate -1.0 into its own category
            binned_col[X_copy[col] == -1.0] = "Missing"
            X_copy[col] = binned_col

        # 2. CALCULATE WOE; IV FOR ALL COLUMNS
        all_features: List[str] = self.numerical_features + self.categorical_features

        total_bad: int = y.sum()
        total_good: int = len(y) - total_bad

        for col in all_features:
            stats_df = pd.DataFrame({"value": X_copy[col], "target": y})

            # Count the number of Bad (1) and Total customer per group
            grouped = stats_df.groupby("value")["target"].agg(
                bad_count="sum", total_count="count"
            )
            grouped["good_count"] = grouped["total_count"] - grouped["bad_count"]

            # Apply smoothing (+0.5) to  prevent dividing by zero errors
            grouped["dist_good"] = (grouped["good_count"] + 0.5) / (total_good + 1)
            grouped["dist_bad"] = (grouped["bad_count"] + 0.5) / (total_bad + 1)

            # Mathematical formula: WoE = ln( %Good / %Bad)
            grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])

            # --- METRIC GENERATION: Compute Information Value (IV) for Feature Selection ---
            grouped["iv_bin"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped[
                "woe"
            ]
            self.iv_scores[col] = float(grouped["iv_bin"].sum())

            # Save the final lookup table
            self.woe_dictionaries[col] = grouped["woe"].to_dict()

        # 3. FEATURE SELECTION (IV & CORRELATION) (Learned from Train set only)
        # Rule: Straightforwardly eliminate features with IV < 0.02.
        IV_THRESHOLD = 0.02
        # Rule 2: Eliminate highly correlated features (Pearson |r| >= 0.7), retaining the one with higher IV.
        CORR_THRESHOLD = 0.7

        # Sifting Mechanics (Phase 1): Retain only features that surpass the regulatory IV threshold
        iv_dropped_features = [
            col for col in all_features if self.iv_scores.get(col, 0.0) < IV_THRESHOLD
        ]
        iv_qualified_features = [
            col for col in all_features if self.iv_scores.get(col, 0.0) >= IV_THRESHOLD
        ]

        # Sifting Mechanics (Phase 2): Detect and resolve multicollinearity using WoE-encoded value
        features_to_drop = set()
        self.corr_drop_details: List[tuple] = []

        if len(iv_qualified_features) > 1:
            # Reconstruct a temporary WoE matrix to calculate correlations
            temp_woe_df = pd.DataFrame()
            for col in iv_qualified_features:
                temp_woe_df[col] = (
                    X_copy[col].astype(str).map(self.woe_dictionaries[col]).fillna(0.0)
                )

            # Calculate the absolute  Pearson correlation matrix
            corr_matrix = temp_woe_df.corr().abs()

            # Scan the lower triangle of the correlation matrix to identify redundant predictors
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    corr_score = corr_matrix.iloc[i, j]
                    if corr_score >= CORR_THRESHOLD:
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]

                        # Retain the predictor with higher Information Value(IV)
                        if self.iv_scores[col1] > self.iv_scores[col2]:
                            dropped, kept = col2, col1
                        else:
                            dropped, kept = col1, col2

                        if dropped not in features_to_drop:
                            features_to_drop.add(dropped)
                            self.corr_drop_details.append((dropped, kept, corr_score))

            # Finalize the structural array by excluding correlated features
            self.selected_features = [
                f for f in iv_qualified_features if f not in features_to_drop
            ]
        else:
            self.selected_features = iv_qualified_features

        # Audit & Traceability: Output pipeline reduction metrics directly to logs
        iv_dropped = len(all_features) - len(iv_qualified_features)
        corr_dropped = len(iv_qualified_features) - len(self.selected_features)

        if iv_dropped_features:
            print(
                f"[FEATURE SELECTION] Dropped {len(iv_dropped_features)} weak features due to IV < {IV_THRESHOLD}:"
            )
            print(f"  -> {iv_dropped_features}")

        if self.corr_drop_details:
            print(
                f"[FEATURE SELECTION] Dropped {len(features_to_drop)} redundant features due to Correlation >= {CORR_THRESHOLD}:"
            )
            for dropped, kept, score in self.corr_drop_details:
                print(
                    f"  -> ❌ [{dropped}] (Dropped due to correlation r = {score:.3f} with '{kept}')"
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the saved bins and WOE scores to a new dataset.

        Args:
            X (pd.DataFrame): The new data to transform.

        Returns:
            pd.DataFrame: A dataset where original values are replaced by WOE numbers.
        """
        X_copy: pd.DataFrame = X.copy()

        # 1. APPLY BINS TO NUMBERS
        for col in self.numerical_features:
            edges = self.bin_edges[col]
            group_names = [f"Bin_{i}" for i in range(len(edges) - 1)]

            binned_col = pd.cut(
                X_copy[col], bins=edges, include_lowest=True, labels=group_names
            ).astype(str)

            # Explicitly separate -1.0 as "Missing" before mapping WOE
            binned_col[X_copy[col] == -1.0] = "Missing"
            binned_col[binned_col == "nan"] = "Missing"  # Catch out-of-bounds NaNs

            X_copy[col] = binned_col

        # 2. MAP VALUES TO WOE SCORES
        all_features: List[str] = self.numerical_features + self.categorical_features
        for col in all_features:
            # Lookup the score in our dictionary: default to 0.0 (neutral) if categorical is unknown
            X_copy[col] = X_copy[col].map(self.woe_dictionaries[col]).fillna(0.0)

        # 3. AUTOMATIC FEATURE SELECTION BASED ON IV THRESHOLD

        # Return the optimized matrix containing only qualified WOE encoded predictors
        return X_copy[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Combines fit() and transform() sequentially

        Args:
            X (pd.DataFrame): Training data
            y (pd.Series): Target labels.

        Returns:
            pd.DataFrame: A dataset transformed into WOE numbers.
        """
        self.fit(X, y)
        return self.transform(X)


# INTERNAL TEST BLOCK
# [MOCK CONFIG] - Mocking src.config to run the file independently
AGE_MIN, AGE_MAX = 18, 100
INCOME_MIN = 0
EMP_LENGTH_MIN, EMP_LENGTH_MAX = 0, 60
LOAN_AMOUNT_MIN = 500
INTEREST_RATE_MIN = 1.0
CREDIT_HISTORY_MIN = 0

if __name__ == "__main__":
    print("--- 🚀 STARTING COMPREHENSIVE PIPELINE TEST 🚀 ---")

    NUM_COLS = ["person_age", "person_income", "loan_amnt", "person_emp_length"]
    # Adding 'loan_percent_income' to test auto-drop and recompute logic
    CAT_COLS = [
        "person_home_ownership",
        "loan_intent",
        "noise_feature",
        "strong_cat_feature",
    ]

    # MOCK DATA (20 rows to ensure qcut works properly)
    sample_X = pd.DataFrame(
        {
            # 1. OUTLIER TEST: Under 18, over 100 -> Converted to NaN -> Imputed as -1.0 -> Separated into "Missing" bin
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
            # 2. MISSING TEST: Natural NaNs -> Imputed as -1.0
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
            # 3. DERIVED FEATURE GENERATION: Calculate loan_percent_income_computed
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
            "loan_percent_income": [0.2]
            * 20,  # Purposely included to test if TIER 1 drops it
            # 4. MULTICOLLINEARITY TEST: Highly correlated with loan_amnt -> Should be dropped in Phase 2
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
            # 5. CAT MISSING & TEXT STANDARDIZATION: Lowercase, whitespaces, NaN -> Uppercase, imputed as "Missing"
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
            # 6. LOW IV TEST: Contains only "A", no predictive power -> IV = 0 -> Should be dropped
            "noise_feature": ["A"] * 20,
            # 7. STRONG IV TEST: High predictive power (Y correlates with Bad=1, X with Good=0) -> Should be kept
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

    # TARGET VARIABLE: 1 = Bad, 0 = Good (Purposely mapped Y=1, X=0 with strong_cat_feature to generate high IV)
    sample_y = pd.Series([0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1])

    # ------------------
    # RUN TIER 1: CLEANING
    # ------------------
    cleaner = CreditDataCleaner(
        numerical_features=NUM_COLS, categorical_features=CAT_COLS
    )
    clean_data = cleaner.fit_transform(sample_X)

    # ------------------
    # RUN TIER 2: WOE & FEATURE SELECTION
    # ------------------
    MOCK_BIN_CONFIG = {
        "default_bins": 3,
        "person_age": 4,  # Try splitting age into 4 bins
    }

    woe_transformer = WOETransformer(
        numerical_features=cleaner.numerical_features,  # Inherit derived features from Tier 1
        categorical_features=cleaner.categorical_features,
        bin_config=MOCK_BIN_CONFIG,
    )

    final_data = woe_transformer.fit_transform(clean_data, sample_y)

    # ------------------
    # TEST UNSEEN DATA (Simulating real inference data)
    # ------------------
    unseen_X = pd.DataFrame(
        {
            "person_age": [25],
            "person_income": [60000],
            "loan_amnt": [10000],
            "person_emp_length": [10],
            "person_home_ownership": [
                "NEW_CATEGORY"
            ],  # Unseen category -> Should fallback to fillna(0.0)
            "loan_intent": ["EDUCATION"],
            "noise_feature": ["B"],
            "strong_cat_feature": ["Z"],
        }
    )
    # Run through the pipeline like real inference
    clean_unseen = cleaner.transform(unseen_X)
    final_unseen = woe_transformer.transform(clean_unseen)

    # ------------------
    # SYSTEM AUDIT LOGS
    # ------------------
    print("\n" + "=" * 50)
    print(" 🛠️ [AUDIT] TIER 1: DATA CLEANER")
    print("=" * 50)
    cleaner._print_audit_log()
    print(
        f"  -> [Check]: Dropped old 'loan_percent_income' column: {'loan_percent_income' not in clean_data.columns}"
    )
    print(
        f"  -> [Check]: Created 'loan_percent_income_computed' column: {'loan_percent_income_computed' in clean_data.columns}"
    )

    print("\n" + "=" * 50)
    print(" 📊 [AUDIT] INFORMATION VALUE (IV) & SELECTION")
    print("=" * 50)
    for feature, iv in sorted(
        woe_transformer.iv_scores.items(), key=lambda x: x[1], reverse=True
    ):
        if feature in woe_transformer.selected_features:
            status = "✅ KEPT"
        elif iv < 0.02:
            status = "❌ DROPPED (Low IV < 0.02)"
        else:
            status = "❌ DROPPED (High Correlation)"

        power = (
            "Uninformative"
            if iv < 0.02
            else "Weak" if iv < 0.1 else "Medium" if iv < 0.3 else "Strong"
        )
        print(f"  - {feature:<30}: IV = {iv:.4f} | Power: {power:<13} | {status}")

    print("\n" + "=" * 50)
    print(" 🔍 [AUDIT] SAMPLE WOE DICTIONARIES (For Strong & Age features)")
    print("=" * 50)
    for feature in ["strong_cat_feature", "person_age"]:
        print(f"\n🔹 Feature: {feature}")
        for category_or_bin, score in woe_transformer.woe_dictionaries[feature].items():
            print(f"    - Bin/Cat: {category_or_bin:<25} | WOE = {score:.4f}")

    print("\n" + "=" * 50)
    print(" 🚀 [AUDIT] UNSEEN DATA INFERENCE TEST")
    print("=" * 50)
    print(
        "Input contains an unknown category ('NEW_CATEGORY'). Does WOE automatically map it to 0.0?"
    )
    print(final_unseen)

    print("\n" + "=" * 50)
    print(" 📑 [AUDIT] DATA MUTATION LOG (DQ-02 COMPLIANCE)")
    print("=" * 50)
    mutation_logs = cleaner.audit_report_.get("mutation_details", [])
    if mutation_logs:
        df_mutations = pd.DataFrame(mutation_logs)
        print(df_mutations.to_string(index=False))
    else:
        print("No mutations required. Original data was clean.")

    print("\n==================================================")
    print("[SUCCESS] Preprocessing Pipeline executed flawlessly! 🎉")
