"""
Credit Scorecard Scaling Engine
------------------------------
Translates raw probability parameters and WOE metrics into standard integer
credit scores using industry-standard financial engineering scaling constants.
Includes automated look-up table generation and visual monotonic audit plots.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List


class CreditScorecardScaler:
    """
    Manages the transformation of Logistic Regression coefficients and WOE mappings
    into individual score weights and scales out global credit scores.
    """

    def __init__(self, scaling_config: Dict[str, Any]) -> None:
        """
        Initializes the scaler engine with business target criteria.

        Args:
            scaling_config (Dict[str, Any]): Dictionary containing base_score, base_odds, and pdo.
        """
        self.base_score: float = float(scaling_config["base_score"])
        self.base_odds: float = float(scaling_config["base_odds"])
        self.pdo: float = float(scaling_config["pdo"])

        # Calculate foundational scaling constants
        self.factor: float = self.pdo / math.log(2)
        self.offset: float = self.base_score - (self.factor * math.log(self.base_odds))

        # Placeholders for generated scorecard parameters
        self.scorecard_table: pd.DataFrame = pd.DataFrame()
        self.points_map: Dict[str, Dict[str, int]] = {}

    def fit(self, model_trainer: Any, woe_transformer: Any) -> "CreditScorecardScaler":
        """
        Processes model coefficients and bin dictionaries to construct the point system.

        Args:
            model_trainer (Any): Fitted CreditModelTrainer instance containing coefficients_ and intercept_.
            woe_transformer (Any): Fitted WOETransformer containing woe_dictionaries and iv_scores.

        Inputs:
            - Model beta weights vector ($\\beta_i$) and intercept ($\\beta_0$).
            - Feature bin limits mappings containing corresponding WOE values.
        Outputs:
            - Populated lookup references maps stored in self.points_map.
        """
        coefficients: Dict[str, float] = model_trainer.coefficients_
        intercept: float = model_trainer.intercept_
        woe_dicts: Dict[str, Dict[str, float]] = woe_transformer.woe_dictionaries

        # Identify features that successfully survived Information Value (IV) filtering
        surviving_features: List[str] = list(coefficients.keys())
        n_features: int = len(surviving_features)

        scorecard_rows: List[Dict[str, Any]] = []
        self.points_map = {}

        # Disperse point weights proportionally across each predictor matrix element
        for feature in surviving_features:
            beta = coefficients[feature]
            feature_woe_dict = woe_dicts[feature]
            self.points_map[feature] = {}

            for bin_name, woe_val in feature_woe_dict.items():
                # Primary credit scaling transformation math execution
                points = -(woe_val * beta * self.factor) + (
                    (self.offset - (intercept * self.factor)) / n_features
                )
                rounded_points = int(round(points))

                self.points_map[feature][bin_name] = rounded_points

                scorecard_rows.append(
                    {
                        "Feature": feature,
                        "Bin/Category": bin_name,
                        "WOE": woe_val,
                        "Beta": beta,
                        "Scaled_Points": rounded_points,
                    }
                )

        self.scorecard_table = pd.DataFrame(scorecard_rows)
        return self

    def transform(self, X_woe_mapped_bins: pd.DataFrame) -> pd.Series:
        """
        Transforms a dataframe containing string bin labels into final integer credit scores.

        Args:
            X_woe_mapped_bins (pd.DataFrame): Dataframe where columns contain text bin strings (e.g., 'Bin_0').

        Inputs:
            - Row-level customer profile string bin attributes matrices.
        Outputs:
            - pd.Series array containing individual customer absolute credit scores.
        """
        score_df = pd.DataFrame(index=X_woe_mapped_bins.index)

        for feature in self.points_map.keys():
            # Map clean bin strings directly to their respective calculated scalar weights
            score_df[feature] = X_woe_mapped_bins[feature].map(self.points_map[feature])

        # Perform aggregate horizontal summation across all predictor elements
        final_credit_scores: pd.Series = score_df.sum(axis=1).astype(int)
        return final_credit_scores

    def export_artifacts(self, folder_path: Path) -> None:
        """
        Saves the lookup database table onto disk storage.

        Args:
            folder_path (Path): Path pointing to the current target run validation storage room.
        """
        folder_path.mkdir(parents=True, exist_ok=True)
        csv_destination = folder_path / "credit_scorecard_lookup_table.csv"
        self.scorecard_table.to_csv(csv_destination, index=False)
        print(
            f"[MLOPS] Financial Look-up Master Scorecard locked at: {csv_destination}"
        )

    def plot_monotonic_barcharts(self, folder_path: Path) -> None:
        """
        Generates analysis bar charts for each feature to verify risk monotonic trends.

        Args:
            folder_path (Path): Destination folder for the visual analysis plot image files.
        """
        folder_path.mkdir(parents=True, exist_ok=True)
        unique_features = self.scorecard_table["Feature"].unique()

        for feature in unique_features:
            feature_data = self.scorecard_table[
                self.scorecard_table["Feature"] == feature
            ].copy()

            # Sort bins sequentially to verify mathematical monotonicity
            feature_data["sort_idx"] = feature_data["Bin/Category"].apply(
                lambda x: int(x.split("_")[1]) if "Bin_" in str(x) else 999
            )
            feature_data = feature_data.sort_values("sort_idx")

            plt.figure(figsize=(8, 5))
            colors = [
                "#2ca02c" if p >= 0 else "#d62728"
                for p in feature_data["Scaled_Points"]
            ]

            bars = plt.bar(
                feature_data["Bin/Category"],
                feature_data["Scaled_Points"],
                color=colors,
                edgecolor="black",
                alpha=0.85,
            )

            # Add value labels above or below each bar for clear lookups
            for bar in bars:
                yval = bar.get_height()
                va_dir = "bottom" if yval >= 0 else "top"
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    yval,
                    f"{int(yval):+d}",
                    ha="center",
                    va=va_dir,
                    fontweight="bold",
                )

            plt.title(
                f"Scorecard Point Distribution: {feature}",
                fontsize=12,
                fontweight="bold",
                pad=15,
            )
            plt.ylabel("Assigned Credit Points", fontsize=10)
            plt.xlabel("Categorical Groups / Numerical Bins", fontsize=10)
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()

            plot_destination = folder_path / f"scorecard_points_{feature}.png"
            plt.savefig(plot_destination, dpi=150)
            plt.close()

        print(
            f"[MLOPS] Successfully rendered {len(unique_features)} monotonic scorecard validation plots."
        )


# ==============================================================================
# INTERNAL TEST BLOCK
# ==============================================================================

if __name__ == "__main__":
    print("--- Testing Scorecard Scaling Engine ---")

    # 1. Define Dummy Mock Business Configuration
    MOCK_SCALING_CONFIG = {"base_score": 600, "base_odds": 50.0, "pdo": 20}

    # 2. Create Mock Objects to simulate fitted upstream pipelines
    class MockModelTrainer:
        def __init__(self):
            self.coefficients_ = {"person_age": -0.8524, "person_income": -1.1256}
            self.intercept_ = 0.4512

    class MockWOETransformer:
        def __init__(self):
            self.iv_scores = {"person_age": 0.4113, "person_income": 0.7093}
            self.woe_dictionaries = {
                "person_age": {"Bin_0": 0.5521, "Bin_1": -0.2145, "Missing": 0.0000},
                "person_income": {
                    "Bin_0": 0.8541,
                    "Bin_1": 0.1245,
                    "Bin_2": -0.6512,
                    "Missing": 0.0000,
                },
            }

    # Initialize components
    mock_model = MockModelTrainer()
    mock_woe = MockWOETransformer()

    # 3. Instantiate and Fit the Scaler Engine
    scaler = CreditScorecardScaler(scaling_config=MOCK_SCALING_CONFIG)
    scaler.fit(model_trainer=mock_model, woe_transformer=mock_woe)

    # 4. Generate Sample Profiles (Mapped string bins) to verify scoring
    sample_X_bins = pd.DataFrame(
        {
            "person_age": ["Bin_0", "Bin_1", "Missing", "Bin_0"],
            "person_income": ["Bin_0", "Bin_1", "Bin_2", "Missing"],
        },
        index=["Customer_A", "Customer_B", "Customer_C", "Customer_D"],
    )

    # Execute mapping transformation
    calculated_scores = scaler.transform(sample_X_bins)

    # ==============================================================================
    # COMPREHENSIVE SYSTEM AUDIT LOGS
    # ==============================================================================
    print("\n==================================================")
    print("[AUDIT] POINT DISTRIBUTION TABLE")
    print("==================================================")
    print(scaler.scorecard_table.to_string(index=False))

    print("\n==================================================")
    print("[AUDIT] SCORING INFERENCE VERIFICATION")
    print("==================================================")
    for client, score in calculated_scores.items():
        print(
            f"  -> Profile Reference: {client:<12} | Final Credit Score = {score} pts"
        )

    print("\n==================================================")
    print("[AUDIT] ARTIFACTS AND GRAPHICAL GENERATION")
    print("==================================================")
    test_run_dir = Path("artifacts/runs/test_run_scorecard")
    scaler.export_artifacts(folder_path=test_run_dir / "tables")
    scaler.plot_monotonic_barcharts(folder_path=test_run_dir / "plots")

    print("\n[SUCCESS] Scorecard module test verification executed seamlessly.")
