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
        """
        coefficients: Dict[str, float] = model_trainer.coefficients_
        intercept: float = model_trainer.intercept_
        woe_dicts: Dict[str, Dict[str, float]] = woe_transformer.woe_dictionaries

        # Identify features that successfully survived Information Value (IV) filtering
        surviving_features: List[str] = list(coefficients.keys())
        n_features: int = len(surviving_features)

        scorecard_rows: List[Dict[str, Any]] = []
        self.points_map = {}

        for feature in surviving_features:
            beta = coefficients[feature]
            feature_woe_dict = woe_dicts[feature]
            self.points_map[feature] = {}

            for bin_name, woe_val in feature_woe_dict.items():
                # Primary credit scaling transformation
                points = -(woe_val * beta * self.factor) + (
                    (self.offset - (intercept * self.factor)) / n_features
                )
                rounded_points = int(round(points))

                # Keep core map keyed on technical bin names to preserve transform pipeline integrity
                self.points_map[feature][bin_name] = rounded_points

                # Smart format intervals to handle both large integers and small ratios
                display_name = bin_name
                if (
                    hasattr(woe_transformer, "bin_edges")
                    and feature in woe_transformer.bin_edges
                ):
                    if bin_name.startswith("Bin_"):
                        try:
                            bin_idx = int(bin_name.split("_")[1])
                            edges = woe_transformer.bin_edges[feature]
                            left_edge = edges[bin_idx]
                            right_edge = edges[bin_idx + 1]

                            # Check if the feature represents a ratio or float component
                            is_float_feature = (
                                any(
                                    isinstance(e, float) and not e.is_integer()
                                    for e in [left_edge, right_edge]
                                )
                                or max(abs(left_edge), abs(right_edge)) <= 1.0
                            )

                            if is_float_feature:
                                display_name = f"[{left_edge:.2f}, {right_edge:.2f})"
                            else:
                                display_name = (
                                    f"[{int(left_edge):,}, {int(right_edge):,})"
                                )
                        except (IndexError, ValueError):
                            pass

                # Establish a stable sorting order to prevent textual sorting mismatches
                sort_idx = 999
                if bin_name.startswith("Bin_"):
                    try:
                        sort_idx = int(bin_name.split("_")[1])
                    except ValueError:
                        pass
                elif bin_name == "Missing":
                    sort_idx = 998

                scorecard_rows.append(
                    {
                        "Feature": feature,
                        "Bin/Category": display_name,
                        "WOE": woe_val,
                        "Beta": beta,
                        "Scaled_Points": rounded_points,
                        "Display_Order": sort_idx,
                    }
                )

        # Cache the original bin edges mapping for downstream artifact serialization
        self.bin_edges = woe_transformer.bin_edges

        # ==============================================================================
        # REGULATORY EXPERT OVERLAY: Enforce economic logic for Credit Bureau flags
        # ==============================================================================
        if "cb_person_default_on_file" in self.points_map:
            pts_n = self.points_map["cb_person_default_on_file"].get("N", 0)
            pts_y = self.points_map["cb_person_default_on_file"].get("Y", 0)

            # If statistical inversion occurs due to multicollinearity, override weights
            if pts_n < pts_y:
                print(
                    f"\n[MLOPS WARNING] Statistical sign reversal detected on 'cb_person_default_on_file'."
                )
                print(
                    f"                Current points: N = {pts_n}, Y = {pts_y} (Violates Risk Intuition)."
                )
                print(
                    f"                Applying Expert Business Overlay to restore risk alignment..."
                )

                # Correct the operational execution lookup maps
                self.points_map["cb_person_default_on_file"]["N"] = pts_y
                self.points_map["cb_person_default_on_file"]["Y"] = pts_n

                # Synchronize the corrections back into the global tracking reporting rows list
                for row in scorecard_rows:
                    if row["Feature"] == "cb_person_default_on_file":
                        if row["Bin/Category"] == "N":
                            row["Scaled_Points"] = pts_y
                        elif row["Bin/Category"] == "Y":
                            row["Scaled_Points"] = pts_n
        # ==============================================================================

        self.scorecard_table = pd.DataFrame(scorecard_rows)
        return self

    def transform(self, X_woe_mapped_bins: pd.DataFrame) -> pd.Series:
        """
        Transforms a dataframe containing string bin labels into final integer credit scores.
        """
        score_df = pd.DataFrame(index=X_woe_mapped_bins.index)

        for feature in self.points_map.keys():
            score_df[feature] = X_woe_mapped_bins[feature].map(self.points_map[feature])

        final_credit_scores: pd.Series = score_df.sum(axis=1).astype(int)
        return final_credit_scores

    def export_artifacts(self, folder_path: Path) -> None:
        """
        Exports both the detailed audit logs, clean business summary tables,
        and production-ready JSON manifests for Streamlit deployment.
        """
        import json

        folder_path.mkdir(parents=True, exist_ok=True)

        # ARTIFACT 1: Detailed Scorecard Audit Trail Table
        audit_destination = folder_path / "credit_scorecard_audit_table.csv"
        self.scorecard_table.to_csv(audit_destination, index=False)
        print(f"[MLOPS] Detailed Scorecard Audit Table locked at: {audit_destination}")

        # ARTIFACT 2: Consolidated Scorecard Business Summary Table
        summary_destination = folder_path / "scorecard_business_summary.csv"
        business_summary = self.scorecard_table[
            ["Feature", "Bin/Category", "Scaled_Points"]
        ].copy()
        business_summary.to_csv(summary_destination, index=False)
        print(
            f"[MLOPS] Clean Business Scorecard Summary Table exported to: {summary_destination}"
        )

        # ARTIFACT 3: Production Deployment Manifest Package for Streamlit
        manifest_destination = folder_path / "scorecard_deployment_manifest.json"

        # Cast numpy float types within bin_edges to native Python floats for JSON serialization
        clean_bin_edges = {}
        if hasattr(self, "bin_edges"):
            for col, edges in self.bin_edges.items():
                clean_bin_edges[col] = [float(x) for x in edges]

        deployment_package = {
            "meta": {
                "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "base_score": self.base_score,
                "pdo": self.pdo,
            },
            "bin_edges": clean_bin_edges,
            "points_map": self.points_map,
        }

        with open(manifest_destination, "w", encoding="utf-8") as f:
            json.dump(deployment_package, f, indent=4, ensure_ascii=False)
        print(
            f"[MLOPS] Streamlit Production Deployment Manifest package generated at: {manifest_destination}"
        )

    def plot_monotonic_barcharts(self, folder_path: Path) -> None:
        """
        Generates analysis bar charts for each feature to verify risk monotonic trends.
        """
        folder_path.mkdir(parents=True, exist_ok=True)
        unique_features = self.scorecard_table["Feature"].unique()

        for feature in unique_features:
            feature_data = self.scorecard_table[
                self.scorecard_table["Feature"] == feature
            ].copy()

            # Sort sequentially using our robust numeric indicator key
            feature_data = feature_data.sort_values("Display_Order")

            plt.figure(
                figsize=(9, 5.5)
            )  # Expanded canvas space to fit descriptive label widths
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

            # Rotate X-axis text ticks by 30 degrees to avoid overlapping labels
            plt.xticks(rotation=30, ha="right", fontsize=9)

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

    MOCK_SCALING_CONFIG = {"base_score": 600, "base_odds": 50.0, "pdo": 20}

    class MockModelTrainer:
        def __init__(self):
            self.coefficients_ = {"person_age": -0.8524, "person_income": -1.1256}
            self.intercept_ = 0.4512

    class MockWOETransformer:
        def __init__(self):
            self.iv_scores = {"person_age": 0.4113, "person_income": 0.7093}
            self.bin_edges = {
                "person_age": [18.0, 30.0, 60.0],
                "person_income": [1000.0, 5000.0, 20000.0],
            }
            self.woe_dictionaries = {
                "person_age": {"Bin_0": 0.5521, "Bin_1": -0.2145, "Missing": 0.0000},
                "person_income": {
                    "Bin_0": 0.8541,
                    "Bin_1": 0.1245,
                    "Bin_2": -0.6512,
                    "Missing": 0.0000,
                },
            }

    mock_model = MockModelTrainer()
    mock_woe = MockWOETransformer()

    scaler = CreditScorecardScaler(scaling_config=MOCK_SCALING_CONFIG)
    scaler.fit(model_trainer=mock_model, woe_transformer=mock_woe)

    sample_X_bins = pd.DataFrame(
        {
            "person_age": ["Bin_0", "Bin_1", "Missing", "Bin_0"],
            "person_income": ["Bin_0", "Bin_1", "Bin_2", "Missing"],
        },
        index=["Customer_A", "Customer_B", "Customer_C", "Customer_D"],
    )

    calculated_scores = scaler.transform(sample_X_bins)

    print("\n==================================================")
    print("[AUDIT] SCORING INFERENCE VERIFICATION")
    print("==================================================")
    for client, score in calculated_scores.items():
        print(
            f"  -> Profile Reference: {client:<12} | Final Credit Score = {score} pts"
        )
