"""
Preprocessing Pipeline for Credit Scoring
-----------------------------------------
This module splits the preprocessing into 2 seperate steps to support different models():
1. CreditDataCleaner: Fill missing values. Used for all models
2. CreditWoETranformer: Converts data to Weight of Evidence (WoE) scores. Used mainly for logistic regression
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# TIER 1: DATA CLEANER


class CreditDataCleaner:
    """
    Step 1: Cleans the data by filling missing values with explicit risk flags.
    """

    def __init__(
        self, numerical_features: List[str], categorical_features: List[str]
    ) -> None:
        """
        Initializes the clenaer with the lists of columns to process.

        Args:
            numerical_features (List[str]):  List of column names containing number.
            categorical_features (List[str]): List of the culumn names containing text.
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "CreditDataCleaner":
        """
        Since we use fixed constants (-1.0 and 'Missing'), there is nothing to learn from the data
        This function exists to maintain a standard pipeline structure.

        Args:
            X (pd.DataFrame): Training data
            y (pd.Series, optional): Target labels.

        Returns:
            CreditDataCleaner: The instance itself
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in the provided dataset.

        Args:
            X (pd.DataFrame): The raw input dataframe.

        Returns:
            pd.DataFrame: A clean dataframe with invalid, duplicate, and missing values handled.
        """
        X_clean: pd.DataFrame = X.copy()

        # Remove duplicate records
        X_clean = X_clean.drop_duplicates()

        # Remove invalid records
        if "person_age" in X_clean.columns:
            X_clean = X_clean[X_clean["person_age"] < 100]

        if "person_emp_length" in X_clean.columns:
            X_clean = X_clean[X_clean["person_emp_length"] < 100]

        # Fill missing numerical values with -1.0
        if self.numerical_features:
            X_clean[self.numerical_features] = X_clean[self.numerical_features].fillna(
                -1.0
            )

        # Fill missing categorical values with "Missing"
        if self.categorical_features:
            X_clean[self.categorical_features] = X_clean[
                self.categorical_features
            ].fillna("Missing")

        return X_clean

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Combines fit() and transform()

        Args:
            X (pd.DataFrame): Training data
            y (pd.Series, optional): Target labels.

        Returns:
            pd.DataFrame: A clean dataframe.
        """
        self.fit(X, y)
        return self.transform(X)


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
        self.numerical_features: List[str] = numerical_features
        self.categorical_features: List[str] = categorical_features
        self.n_bins: Dict[str, int] = bin_config

        # Stores the cut-off points for each numerical column
        self.bin_edges: Dict[str, np.array] = {}

        # Stores the mapping dictionary  from text/bin to WOE score for each column
        self.woe_dictionaries: Dict[str, Dict[str, float]] = {}

        # Stores the IV:
        self.iv_scores: Dict[str, float] = {}
        self.selected_features_: List[str] = (
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

        # 1. FIND BIN EDGES FOR NUMBERS
        for col in self.numerical_features:
            # Exclude the -1.0 missing flag to find true statistical quartiles
            valid_data: pd.Series = X_copy[X_copy[col] != -1.0][col]

            # Update: get number of bin from config.yaml
            n_bins_for_col = self.n_bins.get(col, self.n_bins.get("default_bins", 5))

            _, edges = pd.qcut(
                valid_data, q=n_bins_for_col, retbins=True, duplicates="drop"
            )

            # Extend the lowest boundary to include the -1.0 flag
            if edges[0] > -1.0:
                edges = np.insert(edges, 0, -2.0)

            self.bin_edges[col] = edges

            # Apply bins locally to calculate WoE later in this function
            group_names = [f"Bin_{i}" for i in range(len(edges) - 1)]
            X_copy[col] = pd.cut(
                X_copy[col], bins=edges, include_lowest=True, labels=group_names
            ).astype(str)

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

        # 3. SAVE THE QUALIFIED FEATURE LIST BASED ON IV (Learned from Train set only)
        # Rule: Straightforwardly eliminate features with IV < 0.02.
        IV_THRESHOLD = 0.02
        # Rule 2: Eliminate highly correlated features (Pearson |r| >= 0.7), retaining the one with higher IV.
        CORR_THRESHOLD = 0.7

        # Sifting Mechanics (Phase 1): Retain only features that surpass the regulatory IV threshold
        iv_qualified_features = [
            col for col in all_features if self.iv_scores.get(col, 0) >= IV_THRESHOLD
        ]

        # Sifting Mechanics (Phase 2): Detect and resolve multicollinearity using WoE-encoded value
        if len(iv_qualified_features) > 1:
            # Reconstruct a temporary WoE matrix to calculate correlations
            temp_woe_df = pd.DataFrame()
            for col in iv_qualified_features:
                temp_woe_df[col] = (
                    X_copy[col].astype(str).map(self.woe_dictionaries[col]).fillna(0.0)
                )

            # Calculate the absolute  Pearson correlation matrix
            corr_matrix = temp_woe_df.corr().abs()
            features_to_drop = set()

            # Scan the lower triangle of the correlation matrix to identify redundant predictors
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if corr_matrix.iloc[i, j] >= CORR_THRESHOLD:
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        # Retain the predictor with higher Information Value(IV)
                        if self.iv_scores[col1] > self.iv_scores[col2]:
                            features_to_drop.add(col2)
                        else:
                            features_to_drop.addd(col1)

            # Finalize the structural array by excluding correlated features
            self.selected_features = [
                f for f in iv_qualified_features if f not in features_to_drop
            ]
        else:
            self.selected_features = iv_qualified_features

        # Audit & Traceability: Output pipeline reduction metrics directly to logs
        iv_dropped = len(all_features) - len(iv_qualified_features)
        corr_dropped = len(iv_qualified_features) - len(self.selected_features_)

        if iv_dropped > 0 or corr_dropped > 0:
            print(
                f"[FEATURE SELECTION] Safely dropped {iv_dropped} weak features (IV < {IV_THRESHOLD}) "
                f"and {corr_dropped} redundant features (Correlation >= {CORR_THRESHOLD})"
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
            X_copy[col] = pd.cut(
                X_copy[col], bins=edges, include_lowest=True, labels=group_names
            ).astype(str)
            # Catch unseen out-of-bound data
            X_copy[col] = X_copy[col].replace("nan", "Missing")

        # 2. MAP VALUES TO WOE SCORES
        all_features: List[str] = self.numerical_features + self.categorical_features
        for col in all_features:
            # Lookup the score in our dictionary: default to 0.0 (neutral) if categorical is unknown
            X_copy[col] = X_copy[col].map(self.woe_dictionaries[col]).fillna(0.0)

        # 3. AUTOMATIC FEATURE SELECTION BASED ON IV THRESHOLD

        # Return the optimized matrix containing only qualified WOE encoded predictors
        return X_copy[self.selected_features]

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
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

if __name__ == "__main__":
    print("--- Testing Preprocessing Pipeline ---")

    NUM_COLS = ["person_age", "person_income"]
    CAT_COLS = ["person_home_ownership", "noise_feature"]

    sample_X = pd.DataFrame(
        {
            "person_age": [22, np.nan, 45, 29, 61],
            "person_income": [50000, 85000, np.nan, 42000, 120000],
            "person_home_ownership": ["RENT", "MORTGAGE", "OWN", np.nan, "OWN"],
            "noise_feature": ["A", "A", "A", "A", "A"],
        }
    )
    sample_y = pd.Series([0, 1, 0, 0, 1])  # 1 = Bad, 0 = Good

    # Test Tier 1
    cleaner = CreditDataCleaner(
        numerical_features=NUM_COLS, categorical_features=CAT_COLS
    )
    clean_data = cleaner.fit_transform(sample_X)

    MOCK_BIN_CONFIG = {
        "default_bins": 2,
        "person_age": 2,
        "person_income": 3,  # test
    }

    # Test Tier 2
    woe_transformer = WOETransformer(
        numerical_features=NUM_COLS,
        categorical_features=CAT_COLS,
        bin_config=MOCK_BIN_CONFIG,
    )
    final_data = woe_transformer.fit_transform(clean_data, sample_y)

    # COMPREHENSIVE SYSTEM AUDIT LOGS

    print("\n==================================================")
    print("[AUDIT] WOE ENCODING")
    print("==================================================")
    for feature, woe_dict in woe_transformer.woe_dictionaries.items():
        print(f"\n Feature Column: {feature}")
        for category_or_bin, score in woe_dict.items():
            print(f"    - Sub-group/Bin: {category_or_bin} | WOE Score = {score:.4f}")

    print("[AUDIT] INFORMATION VALUE (IV) SCOREBOARD")
    for feature, iv in woe_transformer.iv_scores.items():
        # Check IV < 0.02
        status = "❌ DROPPED" if iv < 0.02 else "✅ KEPT"
        power = (
            "Uninformative"
            if iv < 0.02
            else "Weak" if iv < 0.1 else "Medium" if iv < 0.3 else "Strong"
        )
        print(
            f"  - {feature:<25}: IV = {iv:.4f} | Power: {power:<13} | Status: {status}"
        )

    print("\n==================================================")
    print("[AUDIT] FINAL MATRICES VALIDATION")
    print("==================================================")
    print(f"Original Input Features : {NUM_COLS + CAT_COLS}")
    print(f"Surviving Output Features: {list(final_data.columns)}")

    print("\n==================================================")
    print("[SUCCESS] Preprocessing Pipeline executed with full visibility.")
