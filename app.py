"""
Streamlit Credit Risk Scorecard Web Application.

This module acts as the user interface layer for the Credit Scorecard System.
It enables interactive single-applicant evaluation, batch CSV scoring, dynamic currency
and income period conversions, Plotly-based score attribution breakdowns (Waterfall Chart),
and underlying scorecard rule inspection.
"""

from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.inference import CreditScorecardInferencePipeline

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CACHING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Scorecard System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_ui_config() -> Dict[str, Any]:
    """Loads and caches user interface localization and layout configurations.

    Returns:
        Dict[str, Any]: Parsed YAML configuration mapping for the UI layout.

    Raises:
        FileNotFoundError: Intercepted via UI alert if ui_config.yaml is missing.
    """
    cfg_path: Path = Path("ui_config.yaml")
    if not cfg_path.exists():
        st.error(f"[CRITICAL] Missing UI config file at: {cfg_path.resolve()}")
        st.stop()

    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_inference_engine() -> CreditScorecardInferencePipeline:
    """Instantiates and caches the core inference pipeline engine in memory.

    Returns:
        CreditScorecardInferencePipeline: Loaded model pipeline ready for scoring.
    """
    return CreditScorecardInferencePipeline()


# Global initialization
ui_cfg: Dict[str, Any] = load_ui_config()
pipeline: CreditScorecardInferencePipeline = get_inference_engine()


# ------------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def format_money(amount: float, curr_unit: str) -> str:
    """Formats monetary values according to currency selection and locale settings.

    Args:
        amount (float): Raw numeric monetary value.
        curr_unit (str): Target currency code ('VND' or 'USD').

    Returns:
        str: Formatted monetary string with localized thousand separators and suffixes.
    """
    if curr_unit == "VND":
        thousand_sep: str = ui_cfg["currency"].get("thousand_separator", ".")
        formatted_str: str = f"{amount:,.0f}".replace(",", thousand_sep)
        return f"{formatted_str} VND"
    else:
        return f"${amount:,.2f} USD"


def render_waterfall_chart(score_breakdown: Dict[str, int], title: str) -> None:
    """Renders an interactive Plotly Waterfall Chart displaying feature point contributions.

    Args:
        score_breakdown (Dict[str, int]): Map of feature names to integer scaled points.
        title (str): Display title for the chart widget.
    """
    features: List[str] = list(score_breakdown.keys())
    values: List[int] = [int(v) for v in score_breakdown.values()]
    total_score: int = sum(values)

    # Append absolute sum total bar at the right end of chart
    x_data: List[str] = features + ["TOTAL SCORE"]
    y_data: List[int] = values + [total_score]
    measures: List[str] = ["relative"] * len(features) + ["total"]
    text_labels: List[str] = [f"{v:+d}" if v != 0 else "0" for v in values] + [
        f"{total_score} pts"
    ]

    fig: go.Figure = go.Figure(
        go.Waterfall(
            name="Credit Score Attribution",
            orientation="v",
            measure=measures,
            x=x_data,
            y=y_data,
            text=text_labels,
            textposition="outside",
            connector={"line": {"color": "#6c757d", "width": 1.5}},
            decreasing={"marker": {"color": "#dc3545"}},
            increasing={"marker": {"color": "#28a745"}},
            totals={"marker": {"color": "#0d6efd"}},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Points Contribution",
        waterfallgap=0.2,
        height=500,
        margin=dict(l=20, r=20, t=50, b=80),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & CONFIGURATION
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title(ui_cfg["project_info"]["title"])
    st.caption(f"🚀 **{ui_cfg['project_info']['team_name']}**")

    logo_path: Path = Path(ui_cfg["project_info"]["logo_path"])
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)

    st.divider()

    lang: str = st.selectbox(
        ui_cfg["i18n"]["en"]["language_select"],
        options=["vi", "en"],
        format_func=lambda x: "Tiếng Việt 🇻🇳" if x == "vi" else "English 🇬🇧",
        index=0,
    )
    t: Dict[str, Any] = ui_cfg["i18n"][lang]

    st.divider()
    st.markdown(f"**🔧 {t['system_status']}**")
    st.info(f"**{t['active_model_run']}:**\n`{pipeline.active_run_dir.name}`")

# ------------------------------------------------------------------------------
# 4. MAIN INTERFACE LAYOUT
# ------------------------------------------------------------------------------
st.title(t["app_title"])
st.caption(ui_cfg["project_info"]["subtitle"])

tab_single, tab_batch = st.tabs([t["single_tab"], t["batch_tab"]])

# ==============================================================================
# TAB 1: SINGLE APPLICANT EVALUATION
# ==============================================================================
with tab_single:
    st.subheader(t["applicant_info"])

    # Currency and income cycle selection controls
    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        currency_unit: str = st.radio(
            f"💵 {t['currency_select']}",
            options=["VND", "USD"],
            horizontal=True,
            index=0,
            key="currency_unit_selector",
        )
    with cfg_col2:
        income_freq: str = st.radio(
            f"📅 {t['income_freq_select']}",
            options=["monthly", "yearly"],
            format_func=lambda x: (
                t["income_freq_monthly"] if x == "monthly" else t["income_freq_yearly"]
            ),
            horizontal=True,
            index=0,
            key="income_freq_selector",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("single_scoring_form"):
        col1, col2, col3 = st.columns(3)

        # Reactive defaults and step increments bound to active currency selection
        default_income: int = 25000000 if currency_unit == "VND" else 1000
        default_loan: int = 50000000 if currency_unit == "VND" else 2000
        step_income: int = 1000000 if currency_unit == "VND" else 100
        step_loan: int = 5000000 if currency_unit == "VND" else 500

        freq_label: str = (
            t["income_freq_monthly"]
            if income_freq == "monthly"
            else t["income_freq_yearly"]
        )

        with col1:
            person_age: int = st.number_input(
                t["person_age"], min_value=18, max_value=100, value=30
            )

            home_opts: List[str] = list(
                ui_cfg["categorical_options"]["person_home_ownership"].keys()
            )
            person_home_ownership: str = st.selectbox(
                t["person_home_ownership"],
                options=home_opts,
                format_func=lambda x: ui_cfg["categorical_options"][
                    "person_home_ownership"
                ][x][lang],
            )

            person_emp_length: float = st.number_input(
                t["person_emp_length"],
                min_value=0.0,
                max_value=60.0,
                value=3.0,
                step=0.5,
            )

        with col2:
            income_input: float = st.number_input(
                f"{t['income_label']} ({freq_label} - {currency_unit})",
                min_value=0,
                value=default_income,
                step=step_income,
            )
            st.caption(
                f"👉 **{format_money(income_input, currency_unit)} / {freq_label.lower()}**"
            )

            intent_opts: List[str] = list(
                ui_cfg["categorical_options"]["loan_intent"].keys()
            )
            loan_intent: str = st.selectbox(
                t["loan_intent"],
                options=intent_opts,
                format_func=lambda x: ui_cfg["categorical_options"]["loan_intent"][x][
                    lang
                ],
            )

            loan_grade: str = st.selectbox(
                t["loan_grade"], options=["A", "B", "C", "D", "E", "F", "G"], index=1
            )

        with col3:
            loan_amt_input: float = st.number_input(
                f"{t['loan_amount_label']} ({currency_unit})",
                min_value=0,
                value=default_loan,
                step=step_loan,
            )
            st.caption(f"👉 **{format_money(loan_amt_input, currency_unit)}**")

            cb_person_cred_hist_length: int = st.number_input(
                t["cb_person_cred_hist_length"], min_value=0, max_value=50, value=5
            )

            cb_default_opts: List[str] = list(
                ui_cfg["categorical_options"]["cb_person_default_on_file"].keys()
            )
            cb_person_default_on_file: str = st.selectbox(
                t["cb_person_default_on_file"],
                options=cb_default_opts,
                format_func=lambda x: ui_cfg["categorical_options"][
                    "cb_person_default_on_file"
                ][x][lang],
            )

            loan_int_rate: float = st.number_input(
                t["loan_int_rate"], min_value=1.0, max_value=40.0, value=11.0, step=0.1
            )

        submit_btn: bool = st.form_submit_button(
            t["btn_predict"], type="primary", use_container_width=True
        )

    # Submission Handler & Evaluation Logic
    if submit_btn:
        usd_rate: float = ui_cfg["currency"].get("usd_to_vnd_rate", 25000.0)

        # 🧮 Step 1: Normalize Income to Annual USD Standard
        if income_freq == "monthly":
            annual_income_raw: float = income_input * 12.0
        else:
            annual_income_raw: float = float(income_input)

        if currency_unit == "VND":
            person_income_usd: float = annual_income_raw / usd_rate
        else:
            person_income_usd: float = annual_income_raw

        # 🧮 Step 2: Normalize Requested Loan Amount to USD
        if currency_unit == "VND":
            loan_amnt_usd: float = loan_amt_input / usd_rate
        else:
            loan_amnt_usd: float = float(loan_amt_input)

        # 🧮 Step 3: Calculate Derived Feature (Debt-to-Income / Loan Percent Income)
        loan_percent_income: float = (
            loan_amnt_usd / person_income_usd if person_income_usd > 0 else 0.0
        )

        # 📦 Step 4: Construct Input DataFrame for Model Inference
        input_data: pd.DataFrame = pd.DataFrame(
            [
                {
                    "person_age": person_age,
                    "person_income": person_income_usd,
                    "person_home_ownership": person_home_ownership,
                    "person_emp_length": person_emp_length,
                    "loan_intent": loan_intent,
                    "loan_grade": loan_grade,
                    "loan_amnt": loan_amnt_usd,
                    "loan_int_rate": loan_int_rate,
                    "loan_percent_income_computed": loan_percent_income,
                    "cb_person_default_on_file": cb_person_default_on_file,
                    "cb_person_cred_hist_length": cb_person_cred_hist_length,
                }
            ]
        )

        try:
            with st.spinner("Processing prediction..."):
                result: Dict[str, Any] = pipeline.predict(input_data)

            st.divider()

            # ==================================================================
            # 1. EVALUATION RESULTS DISPLAY
            # ==================================================================
            st.subheader(t["scoring_result"])

            decision: str = result["decision"]
            score: int = result["credit_score"]
            pd_val: float = result["probability_of_default"]
            risk_band: str = result["risk_band"]

            outcome_cfg: Dict[str, Any] = ui_cfg["mascot"]["outcomes"].get(decision, {})
            mascot_img: str = outcome_cfg.get(
                "image", ui_cfg["mascot"]["default_image"]
            )
            dialogue: str = outcome_cfg.get("dialogue", {}).get(lang, "")
            status_color: str = outcome_cfg.get("status_color", "#6c757d")

            res_col1, res_col2 = st.columns([1, 2.5])

            with res_col1:
                if Path(mascot_img).exists():
                    st.image(mascot_img, use_container_width=True)
                else:
                    st.info("🤖 [Mascot Image]")

            with res_col2:
                st.chat_message("assistant").write(f"**{dialogue}**")
                st.markdown("<br>", unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric(t["score_label"], f"{score} pts")
                m2.metric(t["pd_label"], f"{pd_val * 100:.2f}%")
                m3.metric(t["risk_band_label"], risk_band)

                st.markdown(
                    f"""
                    <div style="background-color: {status_color}; color: white; padding: 12px; border-radius: 8px; text-align: center; font-size: 20px; font-weight: bold; margin-top: 10px;">
                        {t['decision_label']}: {decision}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.divider()

            # ==================================================================
            # 2. WATERFALL SCORE ATTRIBUTION CHART
            # ==================================================================
            score_breakdown: Dict[str, int] = result.get("score_breakdown", {})
            if score_breakdown:
                render_waterfall_chart(score_breakdown, t["waterfall_breakdown"])
            else:
                st.warning(
                    "⚠️ Unable to render Waterfall Chart due to missing breakdown data."
                )

            # ==================================================================
            # 3. SCORECARD RULEBOOK LOOKUP TABLE
            # ==================================================================
            with st.expander(
                "📊 Scorecard Lookup Table & Scaling Factors", expanded=False
            ):
                scaler: Any = pipeline.score_scaler

                if hasattr(scaler, "scorecard_table"):
                    raw_table_str: Any = getattr(scaler, "scorecard_table")
                    if isinstance(raw_table_str, str):
                        df_scorecard: pd.DataFrame = pd.read_csv(
                            StringIO(raw_table_str), sep=r"\s+", engine="python"
                        )
                        st.dataframe(df_scorecard, use_container_width=True)
                    elif isinstance(raw_table_str, pd.DataFrame):
                        st.dataframe(raw_table_str, use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Base Score", getattr(scaler, "base_score", 600))
                c2.metric("Base Odds", getattr(scaler, "base_odds", 50))
                c3.metric("PDO", getattr(scaler, "pdo", 20))
                c4.metric(
                    "Factor / Offset",
                    f"{getattr(scaler, 'factor', 0):.2f} / {getattr(scaler, 'offset', 0):.2f}",
                )

        except Exception as e:
            st.error(f"❌ Execution Error: {e}")

# ==============================================================================
# TAB 2: BATCH APPLICANT SCORING (CSV FILE EVALUATION)
# ==============================================================================
with tab_batch:
    st.subheader(t["batch_header"])
    st.caption(t["batch_caption"])

    # 1. Provide reference CSV template for download
    sample_df: pd.DataFrame = pd.DataFrame(
        [
            {
                "person_age": 30,
                "person_income": 25000000,
                "person_home_ownership": "RENT",
                "person_emp_length": 3.0,
                "loan_intent": "PERSONAL",
                "loan_grade": "B",
                "loan_amnt": 50000000,
                "loan_int_rate": 11.5,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 4,
            },
            {
                "person_age": 45,
                "person_income": 80000000,
                "person_home_ownership": "OWN",
                "person_emp_length": 10.0,
                "loan_intent": "VENTURE",
                "loan_grade": "A",
                "loan_amnt": 100000000,
                "loan_int_rate": 7.5,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 12,
            },
        ]
    )

    col_batch_cfg1, col_batch_cfg2 = st.columns([2, 1])
    with col_batch_cfg2:
        csv_template: bytes = sample_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=t["download_template_btn"],
            data=csv_template,
            file_name="credit_scorecard_batch_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 2. Currency unit selector for batch dataset
    with col_batch_cfg1:
        batch_currency: str = st.radio(
            t["batch_currency_label"],
            options=["VND", "USD"],
            format_func=lambda x: (
                t["batch_currency_vnd"] if x == "VND" else t["batch_currency_usd"]
            ),
            horizontal=True,
            index=0,
            key="batch_currency_selector",
        )

    # 3. CSV File Upload Widget
    uploaded_file: Any = st.file_uploader(
        t["uploader_label"],
        type=["csv"],
        help=t["uploader_help"],
    )

    if uploaded_file is not None:
        try:
            # Parse uploaded CSV file
            raw_batch_df: pd.DataFrame = pd.read_csv(uploaded_file)
            st.success(t["upload_success"].format(count=len(raw_batch_df)))

            with st.expander(t["preview_input_title"], expanded=False):
                st.dataframe(raw_batch_df.head(10), use_container_width=True)

            # Trigger batch scoring pipeline execution
            if st.button(
                t["btn_run_batch"],
                type="primary",
                use_container_width=True,
            ):
                usd_rate: float = ui_cfg["currency"].get("usd_to_vnd_rate", 25000.0)
                batch_input: pd.DataFrame = raw_batch_df.copy()

                # Normalize monetary inputs to USD/Year standard if input is in VND/Monthly
                if batch_currency == "VND":
                    batch_input["person_income"] = (
                        batch_input["person_income"] * 12.0
                    ) / usd_rate
                    batch_input["loan_amnt"] = batch_input["loan_amnt"] / usd_rate

                with st.spinner(t["batch_spinner"]):
                    # Execute prediction pipeline over dataframe
                    batch_results: pd.DataFrame = pipeline.predict(batch_input)

                st.divider()
                st.subheader(t["batch_results_header"])

                # Overall Batch Summary Metrics
                total_records: int = len(batch_results)
                approved_cnt: int = int((batch_results["decision"] == "APPROVED").sum())
                review_cnt: int = int(
                    (batch_results["decision"] == "MANUAL_REVIEW").sum()
                )
                rejected_cnt: int = int((batch_results["decision"] == "REJECTED").sum())

                m1, m2, m3, m4 = st.columns(4)
                m1.metric(t["metric_total"], f"{total_records}")
                m2.metric(
                    t["metric_approved"],
                    f"{approved_cnt} ({approved_cnt/total_records:.1%})",
                )
                m3.metric(
                    t["metric_review"],
                    f"{review_cnt} ({review_cnt/total_records:.1%})",
                )
                m4.metric(
                    t["metric_rejected"],
                    f"{rejected_cnt} ({rejected_cnt/total_records:.1%})",
                )

                # Multiselect Decision Filter Widget
                filter_decision: List[str] = st.multiselect(
                    t["filter_label"],
                    options=["APPROVED", "MANUAL_REVIEW", "REJECTED"],
                    default=["APPROVED", "MANUAL_REVIEW", "REJECTED"],
                )

                filtered_df: pd.DataFrame = batch_results[
                    batch_results["decision"].isin(filter_decision)
                ]

                # Display Results Dataframe
                display_cols: List[str] = [
                    "credit_score",
                    "probability_of_default",
                    "decision",
                    "risk_band",
                    "person_age",
                    "person_income",
                    "loan_amnt",
                ]
                # Fallback in case optional metadata columns are omitted
                available_cols: List[str] = [
                    col for col in display_cols if col in filtered_df.columns
                ]
                st.dataframe(filtered_df[available_cols], use_container_width=True)

                # Export Batch Results to CSV
                result_csv: bytes = batch_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=t["export_btn"],
                    data=result_csv,
                    file_name="credit_scoring_batch_results.csv",
                    mime="text/csv",
                    type="primary",
                )

        except Exception as e:
            st.error(t["error_csv"].format(error=e))
