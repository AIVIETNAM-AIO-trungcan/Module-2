"""
Streamlit Credit Risk Scorecard Web Application.

This module acts as the user interface layer for the Credit Scorecard System.
Refactored to meet banking UI/UX compliance standards, streamline batch processing,
enforce internal underwriter security controls, maintain session state persistence,
dynamically fetch decision thresholds, and ensure Notebook Sync Verification.
Includes an Interactive Decision Simulator with Heatmap formatting,
dynamic Chart visualization for the Scorecard Rulebook, and categorized input forms.
"""

from io import StringIO
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yaml
from huggingface_hub import hf_hub_download

from src.inference import CreditScorecardInferencePipeline

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & AUTOMATIC ASSET SYNC
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Scorecard System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


def force_ensure_assets() -> None:
    """
    Ensures the asset directory exists and contains necessary mascot images.
    Downloads asset.zip from the Hugging Face Hub if missing during Cloud deployment.
    """
    project_root = Path.cwd()
    asset_dir = project_root / "asset"
    asset_dir.mkdir(parents=True, exist_ok=True)

    main_logo = asset_dir / "capybara_main.png"
    if not main_logo.exists():
        try:
            zip_path = hf_hub_download(
                repo_id="trungcan94/AIO_moddule_2_model",
                filename="asset.zip",
                repo_type="dataset",
            )
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(asset_dir)
            print(
                f"[ASSET SYNC] Extracted asset files into asset_dir: {os.listdir(asset_dir)}"
            )
        except Exception as e:
            print(f"[ASSET SYNC ERROR] Failed to fetch assets from Hugging Face: {e}")


# Execute asset verification on initial load
force_ensure_assets()


@st.cache_data
def load_ui_config() -> Dict[str, Any]:
    """
    Loads and caches user interface localization and layout configurations.

    Returns:
        Dict[str, Any]: Parsed YAML configuration dictionary for the UI layer.

    Raises:
        FileNotFoundError: Triggered if ui_config.yaml is missing from the root directory.
    """
    cfg_path: Path = Path("ui_config.yaml")
    if not cfg_path.exists():
        st.error(f"[CRITICAL FAILURE] Missing UI config file at: {cfg_path.resolve()}")
        st.stop()

    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_inference_engine() -> CreditScorecardInferencePipeline:
    """
    Instantiates and caches the core inference pipeline engine in memory.

    Returns:
        CreditScorecardInferencePipeline: Loaded model pipeline ready for real-time scoring.
    """
    return CreditScorecardInferencePipeline()


# Global initialization of UI Configurations and Inference Pipeline
ui_cfg: Dict[str, Any] = load_ui_config()
pipeline: CreditScorecardInferencePipeline = get_inference_engine()


# ------------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def format_money(amount: float, curr_unit: str) -> str:
    """
    Formats monetary values according to the selected currency and locale settings.

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
    """
    Renders an interactive Plotly Waterfall Chart displaying feature point contributions.

    Args:
        score_breakdown (Dict[str, int]): Map of feature names to their respective integer points.
        title (str): Display title for the chart widget.
    """
    features: List[str] = list(score_breakdown.keys())
    values: List[int] = [int(v) for v in score_breakdown.values()]
    total_score: int = sum(values)

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
        xaxis_title="Feature Components",
        yaxis_title="Points Contribution",
        waterfallgap=0.2,
        height=450,
        margin=dict(l=20, r=20, t=50, b=80),
        xaxis=dict(tickangle=-25),
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & DYNAMIC CONFIGURATION
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title(ui_cfg["project_info"]["title"])
    st.caption(f"🚀 **{ui_cfg['project_info']['team_name']}**")

    # Load and display dynamic mascot logo
    logo_filename = Path(ui_cfg["project_info"]["logo_path"]).name
    logo_path = Path.cwd() / "asset" / logo_filename

    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)

    st.divider()

    lang: str = st.selectbox(
        ui_cfg["i18n"]["en"]["language_select"],
        options=["vi", "en"],
        format_func=lambda x: "Tiếng Việt 🇻🇳" if x == "vi" else "English 🇬🇧",
        index=1,
    )
    t: Dict[str, Any] = ui_cfg["i18n"][lang]

    st.divider()

    # Dynamic System Diagnostics and Cut-off Thresholds Display
    st.markdown(f"**🔧 {t['system_status']}**")
    st.info(f"**{t['active_model_run']}:**\n`{pipeline.active_run_dir.name}`")

    # Extract dynamic scoring thresholds from the backend pipeline configuration
    approval_score = pipeline.config.get("scoring_thresholds", {}).get(
        "approval_score", 533
    )
    review_score = pipeline.config.get("scoring_thresholds", {}).get(
        "review_score", 503
    )

    st.markdown("**⚖️ Decision Thresholds (Auto-Synced)**")
    st.success(f"🟢 **Auto-Approve:** $\\geq$ {approval_score} pts")
    st.warning(f"🟡 **Manual Review:** {review_score} to {approval_score - 1} pts")
    st.error(f"🔴 **Reject:** $<$ {review_score} pts")

# ------------------------------------------------------------------------------
# 4. MAIN INTERFACE LAYOUT & TAB LOGIC
# ------------------------------------------------------------------------------
st.title(t["app_title"])
st.caption(ui_cfg["project_info"]["subtitle"])

tab_single, tab_batch = st.tabs([t["single_tab"], t["batch_tab"]])

# ==============================================================================
# TAB 1: SINGLE APPLICANT EVALUATION
# ==============================================================================
with tab_single:
    st.subheader(t["applicant_info"])

    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        currency_unit: str = st.radio(
            f"💵 {t['currency_select']}",
            options=["VND", "USD"],
            horizontal=True,
            index=1,
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
            index=1,
            key="income_freq_selector",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("single_scoring_form"):

        default_income: int = 25000000 if currency_unit == "VND" else 1000
        default_loan: int = 50000000 if currency_unit == "VND" else 2000
        step_income: int = 1000000 if currency_unit == "VND" else 100
        step_loan: int = 5000000 if currency_unit == "VND" else 500

        freq_label: str = (
            t["income_freq_monthly"]
            if income_freq == "monthly"
            else t["income_freq_yearly"]
        )

        # --- Section 1: Applicant Demographics & History ---
        sec1_title = (
            "👤 Thông tin Khách hàng" if lang == "vi" else "👤 Applicant Information"
        )
        st.markdown(f"**{sec1_title}**")

        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            person_age: int = st.number_input(
                t["person_age"], min_value=18, max_value=100, value=30
            )

            person_emp_length: float = st.number_input(
                t["person_emp_length"],
                min_value=0.0,
                max_value=60.0,
                value=3.0,
                step=0.5,
            )

        with col_a2:
            income_input: float = st.number_input(
                f"{t['income_label']} ({freq_label} - {currency_unit})",
                min_value=0.0,
                value=float(default_income),
                step=float(step_income),
            )
            st.caption(
                f"👉 **{format_money(income_input, currency_unit)} / {freq_label.lower()}**"
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

        with col_a3:
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

        st.markdown("<hr style='margin: 15px 0 20px 0;'>", unsafe_allow_html=True)

        # --- Section 2: Loan Information ---
        sec2_title = "💰 Thông tin Khoản vay" if lang == "vi" else "💰 Loan Information"
        st.markdown(f"**{sec2_title}**")

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            loan_amt_input: float = st.number_input(
                f"{t['loan_amount_label']} ({currency_unit})",
                min_value=0.0,
                value=float(default_loan),
                step=float(step_loan),
            )
            st.caption(f"👉 **{format_money(loan_amt_input, currency_unit)}**")

        with col_l2:
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

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn: bool = st.form_submit_button(
            t["btn_predict"], type="primary", use_container_width=True
        )

    # Trigger Single Record Prediction Process
    if submit_btn:
        usd_rate: float = ui_cfg["currency"].get("usd_to_vnd_rate", 25000.0)

        # Normalize Input Income to Annual USD Base Standard
        if income_freq == "monthly":
            annual_income_raw: float = income_input * 12.0
        else:
            annual_income_raw: float = float(income_input)

        person_income_usd: float = (
            annual_income_raw / usd_rate
            if currency_unit == "VND"
            else annual_income_raw
        )

        # Normalize Requested Loan Amount to USD Base Standard
        loan_amnt_usd: float = (
            loan_amt_input / usd_rate
            if currency_unit == "VND"
            else float(loan_amt_input)
        )

        # Compute Debt-to-Income Ratio
        loan_percent_income: float = (
            loan_amnt_usd / person_income_usd if person_income_usd > 0 else 0.0
        )

        # Construct payload dataframe mirroring required schema
        # Injecting missing/NaN values for obsolete post-decision features to bypass validation safely
        input_data: pd.DataFrame = pd.DataFrame(
            [
                {
                    "person_age": person_age,
                    "person_income": person_income_usd,
                    "person_home_ownership": person_home_ownership,
                    "person_emp_length": person_emp_length,
                    "loan_intent": loan_intent,
                    "loan_amnt": loan_amnt_usd,
                    "loan_percent_income": loan_percent_income,
                    "cb_person_default_on_file": cb_person_default_on_file,
                    "cb_person_cred_hist_length": cb_person_cred_hist_length,
                    "loan_grade": np.nan,  # Explicitly mark as missing
                    "loan_int_rate": np.nan,  # Explicitly mark as missing
                }
            ]
        )

        try:
            with st.spinner("Processing prediction via Core Pipeline..."):
                st.session_state["scoring_result"] = pipeline.predict(input_data)
        except Exception as e:
            st.error(f"❌ Execution Error: {e}")

    # Display the Evaluation Panel if scoring session state is populated
    if "scoring_result" in st.session_state:
        result: Dict[str, Any] = st.session_state["scoring_result"]

        st.divider()

        # ------------------------------------------------------------------
        # A. PUBLIC EVALUATION RESULTS & MASCOT DECISION DISPLAY
        # ------------------------------------------------------------------
        st.subheader(t["scoring_result"])

        decision: str = result["decision"]
        score: int = result["credit_score"]
        pd_val: float = result["probability_of_default"]
        risk_band: str = result["risk_band"]

        outcome_cfg: Dict[str, Any] = ui_cfg["mascot"]["outcomes"].get(decision, {})
        mascot_img_path: str = outcome_cfg.get(
            "image", ui_cfg["mascot"]["default_image"]
        )
        dialogue: str = outcome_cfg.get("dialogue", {}).get(lang, "")
        status_color: str = outcome_cfg.get("status_color", "#6c757d")

        res_col1, res_col2 = st.columns([1, 2.5])

        with res_col1:
            mascot_filename = Path(mascot_img_path).name
            mascot_path = Path.cwd() / "asset" / mascot_filename

            if mascot_path.exists():
                st.image(str(mascot_path), use_container_width=True)
            else:
                st.info("🤖 [Mascot Image Loading...]")

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

        # ------------------------------------------------------------------
        # B. INTERNAL UNDERWRITER PORTAL (PASSCODE-PROTECTED RESTRICTIONS)
        # ------------------------------------------------------------------
        st.subheader(t["admin_lock_title"])
        st.caption(t["admin_lock_caption"])

        if "admin_authenticated" not in st.session_state:
            st.session_state["admin_authenticated"] = False

        if not st.session_state["admin_authenticated"]:
            with st.expander("🔑 " + t["admin_pass_label"], expanded=True):
                admin_pass_input = st.text_input(
                    t["admin_pass_label"],
                    type="password",
                    placeholder=t["admin_pass_placeholder"],
                    key="admin_pass_field",
                )
                unlock_btn = st.button(t["admin_unlock_btn"], type="secondary")

                if unlock_btn:
                    correct_pass = str(
                        ui_cfg.get("admin_security", {}).get("passcode", "admin123")
                    )
                    if admin_pass_input == correct_pass:
                        st.session_state["admin_authenticated"] = True
                        st.rerun()
                    else:
                        st.error(t["admin_auth_error"])

        if st.session_state["admin_authenticated"]:
            st.success("🟢 " + t["admin_auth_success"])

            # Render Waterfall Score Attribution Chart
            score_breakdown: Dict[str, int] = result.get("score_breakdown", {})
            if score_breakdown:
                render_waterfall_chart(score_breakdown, t["waterfall_breakdown"])

            # Render Detailed Scorecard Rulebook Visualization
            with st.expander(
                "📊 Banking Scorecard Rulebook & Scaling Parameters (Notebook Sync Verification)",
                expanded=True,
            ):
                scaler: Any = pipeline.score_scaler

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Base Score", getattr(scaler, "base_score", 600))
                c2.metric("Base Odds", getattr(scaler, "base_odds", 50))
                c3.metric("PDO", getattr(scaler, "pdo", 20))
                c4.metric(
                    "Factor / Offset",
                    f"{getattr(scaler, 'factor', 0):.2f} / {getattr(scaler, 'offset', 0):.2f}",
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # Transform Scorecard Table into Intuitive Feature-Level Bar Charts
                if hasattr(scaler, "scorecard_table") and isinstance(
                    scaler.scorecard_table, pd.DataFrame
                ):
                    business_table = scaler.scorecard_table[
                        ["Feature", "Bin/Category", "Scaled_Points"]
                    ].copy()

                    # 1. Filter out Missing/Special bins if they carry 0 points (Reduce Noise)
                    mask_to_drop = business_table["Bin/Category"].astype(str).isin(
                        ["Missing", "Special"]
                    ) & (business_table["Scaled_Points"] == 0)
                    df_plot = business_table[~mask_to_drop].copy()

                    # 2. Sort to ensure logical progression of points
                    df_plot = df_plot.sort_values(by=["Feature", "Scaled_Points"])

                    st.markdown("##### 📈 Feature-Level Point Allocations")

                    # 3. Create a 2-column grid for separated feature charts
                    chart_cols = st.columns(2)
                    unique_features = df_plot["Feature"].unique()

                    for idx, feature_name in enumerate(unique_features):
                        feat_df = df_plot[df_plot["Feature"] == feature_name]

                        fig_feat = px.bar(
                            feat_df,
                            x="Scaled_Points",
                            y="Bin/Category",
                            orientation="h",
                            text="Scaled_Points",
                            color="Scaled_Points",
                            color_continuous_scale="RdYlGn",
                            title=f"Feature: <b>{feature_name}</b>",
                            labels={"Scaled_Points": "Points", "Bin/Category": "Bin"},
                        )

                        # Dynamic height based on bins to prevent squishing
                        fig_feat.update_layout(
                            height=150 + len(feat_df) * 35,
                            showlegend=False,
                            margin=dict(l=10, r=30, t=40, b=20),
                            yaxis={
                                "categoryorder": "array",
                                "categoryarray": feat_df["Bin/Category"],
                            },
                            coloraxis_showscale=False,  # Hide colorbar to save space
                        )
                        fig_feat.update_traces(textposition="auto")

                        chart_cols[idx % 2].plotly_chart(
                            fig_feat, use_container_width=True
                        )

# ==============================================================================
# TAB 2: BATCH APPLICANT SCORING (CSV FILE EVALUATION)
# ==============================================================================
with tab_batch:
    st.subheader(t["batch_header"])
    st.caption(t["batch_caption"])

    # Create template reflecting the actual necessary inputs (without post-decision leakage features)
    sample_df: pd.DataFrame = pd.DataFrame(
        [
            {
                "person_age": 30,
                "person_income": 25000,
                "person_home_ownership": "RENT",
                "person_emp_length": 3.0,
                "loan_intent": "PERSONAL",
                "loan_amnt": 5000,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 4,
            },
            {
                "person_age": 45,
                "person_income": 80000,
                "person_home_ownership": "OWN",
                "person_emp_length": 10.0,
                "loan_intent": "VENTURE",
                "loan_amnt": 15000,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 12,
            },
        ]
    )

    csv_template: bytes = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=t["download_template_btn"],
        data=csv_template,
        file_name="credit_scorecard_batch_template.csv",
        mime="text/csv",
        use_container_width=False,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file: Any = st.file_uploader(
        t["uploader_label"],
        type=["csv"],
        help=t["uploader_help"],
    )

    if uploaded_file is not None:
        try:
            raw_batch_df: pd.DataFrame = pd.read_csv(uploaded_file)
            st.success(t["upload_success"].format(count=len(raw_batch_df)))

            with st.expander(t["preview_input_title"], expanded=False):
                st.dataframe(raw_batch_df.head(10), use_container_width=True)

            if st.button(
                t["btn_run_batch"],
                type="primary",
                use_container_width=True,
            ):
                batch_input: pd.DataFrame = raw_batch_df.copy()

                # Automatically compute 'loan_percent_income' if missing from the uploaded CSV
                if "loan_percent_income" not in batch_input.columns:
                    if (
                        "loan_amnt" in batch_input.columns
                        and "person_income" in batch_input.columns
                    ):
                        batch_input["loan_percent_income"] = np.where(
                            batch_input["person_income"] > 0,
                            batch_input["loan_amnt"] / batch_input["person_income"],
                            0.0,
                        ).round(4)

                # Inject missing/NaN values for legacy post-decision features to bypass strict engine validation
                if "loan_grade" not in batch_input.columns:
                    batch_input["loan_grade"] = np.nan
                if "loan_int_rate" not in batch_input.columns:
                    batch_input["loan_int_rate"] = np.nan

                with st.spinner(t["batch_spinner"]):
                    # Predict batch and cache to session state to prevent re-execution
                    st.session_state["batch_results"] = pipeline.predict(batch_input)

            # Render Interactive Decision Simulator
            if "batch_results" in st.session_state:
                batch_results: pd.DataFrame = st.session_state["batch_results"]

                st.divider()
                sim_title = (
                    "🎛️ Trình giả lập Chiến lược (Interactive Decision Simulator)"
                    if lang == "vi"
                    else "🎛️ Interactive Decision Strategy Simulator"
                )
                st.subheader(sim_title)

                # Fetch default bounds from pipeline config
                default_app = int(
                    pipeline.config.get("scoring_thresholds", {}).get(
                        "approval_score", 533
                    )
                )
                default_rev = int(
                    pipeline.config.get("scoring_thresholds", {}).get(
                        "review_score", 503
                    )
                )

                slider_label = (
                    "Điều chỉnh Ngưỡng ra Quyết định (Duyệt thủ công & Tự động duyệt):"
                    if lang == "vi"
                    else "Adjust Decision Thresholds (Manual Review & Auto-Approve):"
                )

                # Interactive Double-Ended Slider
                rev_thresh, app_thresh = st.slider(
                    slider_label,
                    min_value=300,
                    max_value=850,
                    value=(default_rev, default_app),
                    step=1,
                )

                # Dynamic Decision Recalculation
                def simulate_decision(score):
                    if score >= app_thresh:
                        return "APPROVED"
                    elif score >= rev_thresh:
                        return "MANUAL_REVIEW"
                    else:
                        return "REJECTED"

                sim_df = batch_results.copy()
                sim_df["simulated_decision"] = sim_df["credit_score"].apply(
                    simulate_decision
                )

                # Extract dynamic metrics
                app_mask = sim_df["simulated_decision"] == "APPROVED"
                rev_mask = sim_df["simulated_decision"] == "MANUAL_REVIEW"
                rej_mask = sim_df["simulated_decision"] == "REJECTED"

                total_records: int = len(sim_df)
                approved_cnt: int = int(app_mask.sum())
                review_cnt: int = int(rev_mask.sum())
                rejected_cnt: int = int(rej_mask.sum())

                # Calculate Expected Bad Rate for the Approved cohort
                expected_bad_rate = (
                    sim_df.loc[app_mask, "probability_of_default"].mean()
                    if approved_cnt > 0
                    else 0.0
                )

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric(t["metric_total"], f"{total_records}")
                m2.metric(
                    "🟢 " + (t.get("metric_approved", "Approved").replace("🟢 ", "")),
                    f"{approved_cnt} ({approved_cnt/total_records:.1%})",
                )
                m3.metric(
                    "🟡 " + (t.get("metric_review", "Review").replace("🟡 ", "")),
                    f"{review_cnt} ({review_cnt/total_records:.1%})",
                )
                m4.metric(
                    "🔴 " + (t.get("metric_rejected", "Rejected").replace("🔴 ", "")),
                    f"{rejected_cnt} ({rejected_cnt/total_records:.1%})",
                )

                bad_rate_label = (
                    "⚠️ Nợ xấu dự kiến" if lang == "vi" else "⚠️ Expected Bad Rate"
                )
                m5.metric(bad_rate_label, f"{expected_bad_rate:.2%}")

                # --------------------------------------------------------------
                # CONDITIONAL FORMATTING HEATMAP (PANDAS STYLER)
                # --------------------------------------------------------------
                filter_label = "Lọc kết quả:" if lang == "vi" else "Filter results:"
                filter_decision: List[str] = st.multiselect(
                    filter_label,
                    options=["APPROVED", "MANUAL_REVIEW", "REJECTED"],
                    default=["APPROVED", "MANUAL_REVIEW", "REJECTED"],
                )

                filtered_df: pd.DataFrame = sim_df[
                    sim_df["simulated_decision"].isin(filter_decision)
                ]

                display_cols: List[str] = [
                    "credit_score",
                    "probability_of_default",
                    "simulated_decision",
                    "risk_band",
                    "person_age",
                    "person_income",
                    "loan_amnt",
                ]
                available_cols: List[str] = [
                    col for col in display_cols if col in filtered_df.columns
                ]
                display_df = filtered_df[available_cols].copy()

                # DataFrame Heatmap Styling Strategy
                def style_dataframe(df):
                    def color_decision(val):
                        if val == "APPROVED":
                            return "color: #155724; background-color: #d4edda; font-weight: bold;"
                        elif val == "REJECTED":
                            return "color: #721c24; background-color: #f8d7da; font-weight: bold;"
                        elif val == "MANUAL_REVIEW":
                            return "color: #856404; background-color: #fff3cd; font-weight: bold;"
                        return ""

                    return (
                        df.style.map(color_decision, subset=["simulated_decision"])
                        .background_gradient(
                            cmap="RdYlGn_r", subset=["probability_of_default"]
                        )
                        .background_gradient(cmap="RdYlGn", subset=["credit_score"])
                        .format({"probability_of_default": "{:.2%}"})
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(style_dataframe(display_df), use_container_width=True)

                result_csv: bytes = sim_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=t["export_btn"],
                    data=result_csv,
                    file_name="credit_scoring_batch_simulated_results.csv",
                    mime="text/csv",
                    type="primary",
                )

        except Exception as e:
            st.error(t["error_csv"].format(error=e))
