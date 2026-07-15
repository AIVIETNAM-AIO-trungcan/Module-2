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
        This function exitss tp maintain a standard pipeline structure.

        Args:
            X (pd.DataFrame): Training data
            y (pd.Series, optional): Target labels.

        Returns:
            CreditDataCleaner: The instance itself
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in the provides dataset.

        Args:
            X (pd.DataFrame): the raw input dataframe.

        Returns:
            pd.DataFrame: A clean dataframe with no NaN values.
        """
        X_clean: pd.DataFrame = X.copy()

        # Fill missing numbers with -1.0
        if self.numerical_features:
            X_clean[self.numerical_features] = X_clean[self.numerical_features].fillna(
                -1.0
            )

        # Fill missing text with "Missing"
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
        n_bins: int = 5,
    ) -> None:
        """
        Initializes the WOE transformer.

        Args:
            numerical_features (List[str]): Numerical columns to be binned.
            categorical_features (List[str]): Text columns to be encoded.
            n_bins (int): The number of intervals to split numerical data into.
        """
        self.numerical_features: List[str] = numerical_features
        self.categorical_features: List[str] = categorical_features
        self.n_bins: int = n_bins

        # Stores the cut-off points for each numerical column
        self.bin_edges: Dict[str, np.array] = {}

        # Stores the mapping dictionary  from text/bin to WOE score for each column
        self.woe_dictionaries: Dict[str, Dict[str, float]] = {}

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
            _, edges = pd.qcut(
                valid_data, q=self.n_bins, retbins=True, duplicates="drop"
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

        # 2. CALCULATE WOE FOR ALL COLUMNS
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

            # Save the final lookup table
            self.woe_dictionaries[col] = grouped["woe"].to_dict()

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

        return X_copy

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
    CAT_COLS = ["person_home_ownership"]

    sample_X = pd.DataFrame(
        {
            "person_age": [22, np.nan, 45, 29, 61],
            "person_income": [50000, 85000, np.nan, 42000, 120000],
            "person_home_ownership": ["RENT", "MORTGAGE", "OWN", np.nan, "OWN"],
        }
    )
    sample_y = pd.Series([0, 1, 0, 0, 1])  # 1 = Bad, 0 = Good

    # Test Tier 1
    cleaner = CreditDataCleaner(
        numerical_features=NUM_COLS, categorical_features=CAT_COLS
    )
    clean_data = cleaner.fit_transform(sample_X)

    # Test Tier 2
    woe_transformer = WOETransformer(
        numerical_features=NUM_COLS, categorical_features=CAT_COLS, n_bins=2
    )
    final_data = woe_transformer.fit_transform(clean_data, sample_y)

    print("\n[Audit] WOE Dictionary for 'person_home_ownership':")
    for category, score in woe_transformer.woe_dictionaries[
        "person_home_ownership"
    ].items():
        print(f"  - {category}: {score:.4f}")

    print("\n[SUCCESS] Pipeline executed")
