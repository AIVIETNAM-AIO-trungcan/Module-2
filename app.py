import streamlit as st
import pandas as pd
import joblib
import os
import glob
import time

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Credit Risk Assessment System", page_icon="🏦", layout="wide"
)

# 2. DICTIONARY FOR MULTI-LANGUAGE SUPPORT & PROFESSIONAL UI LABELS
I18N = {
    "VN": {
        "title": "🏦 Hệ Thống Thẩm Định & Chấm Điểm Tín Dụng Tự Động",
        "subtitle": "Cổng tiếp nhận và phân tích hồ sơ vay vốn tự động theo tiêu chuẩn quản trị rủi ro ngân hàng.",
        "sidebar_title": "📋 Đăng Ký Hồ Sơ Vay Vốn",
        "engine_label": "Phiên bản mô hình:",
        "sec_personal": "👤 I. Thông Tin Cá Nhân & Tài Chính",
        "sec_loan": "💳 II. Chi Tiết Khoản Vay & Tín Dụng",
        # Field Labels & Helps
        "age": "Độ tuổi người vay",
        "age_help": "Độ tuổi hợp lệ từ 18 đến 100 tuổi.",
        "income": "Thu nhập bình quân hàng năm ($)",
        "income_help": "Tổng thu nhập trước thuế từ lương và các nguồn hợp pháp trong 1 năm.",
        "emp_length": "Thâm niên làm việc (năm)",
        "emp_length_help": "Tổng thời gian công tác tích lũy tại đơn vị hiện tại hoặc liên tục.",
        "loan_amnt": "Số tiền đề nghị vay ($)",
        "loan_amnt_help": "Tổng hạn mức tín dụng khách hàng muốn đăng ký vay.",
        "loan_int_rate": "Lãi suất dự kiến (%/năm)",
        "loan_int_rate_help": "Lãi suất áp dụng cho gói vay tương ứng.",
        "cred_hist_len": "Thâm niên lịch sử tín dụng (năm)",
        "cred_hist_len_help": "Số năm khách hàng bắt đầu mở tài khoản tín dụng/khoản vay đầu tiên.",
        "home_ownership": "Hình thức sở hữu nhà ở",
        "home_ownership_help": "Tình trạng pháp lý về nơi ở hiện tại của khách hàng.",
        "loan_intent": "Mục đích sử dụng vốn vay",
        "loan_intent_help": "Lĩnh vực hoặc nhu cầu tài chính thực tế cần giải ngân.",
        "loan_grade": "Phân hạng rủi ro khoản vay",
        "loan_grade_help": "Hạng tín dụng từ A (Rủi ro thấp nhất) đến G (Rủi ro cao nhất).",
        "default_on_file": "Ghi nhận lịch sử quá hạn/vỡ nợ",
        "default_on_file_help": "Khách hàng từng có nợ quá hạn hoặc ghi nhận vỡ nợ trên CIC hay chưa.",
        # Buttons & Messages
        "confirm_check": "Tôi xin cam đoan các thông tin kê khai trên là hoàn toàn chính xác và chịu trách nhiệm trước pháp luật.",
        "submit_btn": "🚀 Gửi Hồ Sơ & Thẩm Định",
        "guide_info": "👈 Vui lòng điền đầy đủ thông tin hồ sơ bên thanh Sidebar và nhấn **'Gửi Hồ Sơ & Thẩm Định'** để bắt đầu.",
        "warn_confirm": "⚠️ Vui lòng đánh dấu xác nhận cam đoan thông tin chính xác trước khi gửi hồ sơ.",
        "spinner_msg": "🔄 Hệ thống AI đang phân tích hồ sơ và tính toán xác suất vỡ nợ... Vui lòng đợi trong giây lát.",
        "res_header": "📊 Kết Quả Thẩm Định Chi Tiết",
        "kpi_score": "Điểm Tín Dụng (Credit Score)",
        "kpi_pd": "Xác Suất Vỡ Nợ (PD)",
        "kpi_ratio": "Tỷ lệ Vay / Thu nhập",
        "dec_header": "🏆 Quyết Định Phê Duyệt Tín Dụng",
        # Approval Scenarios
        "approved_title": "✅ HỒ SƠ ĐƯỢC PHÊ DUYỆT TỰ ĐỘNG (APPROVED)",
        "approved_msg": "Chúc mừng! Điểm tín dụng của bạn đạt mức **{score} điểm**. Khoản vay trị giá **${amount:,.0f}** đã được hệ thống tự động phê duyệt.",
        "pending_title": "⚠️ HỒ SƠ CẦN THẨM ĐỊNH BỔ SUNG (PENDING REVIEW)",
        "pending_msg": "Điểm tín dụng của bạn đạt mức **{score} điểm**. Để hoàn tất thủ tục giải ngân, bộ phận Quản trị Rủi ro cần xác minh thêm một số chứng từ.",
        "contact_form_title": "**Vui lòng để lại thông tin để Chuyên viên Tín dụng liên hệ hỗ trợ trực tiếp:**",
        "email_label": "Địa chỉ Email liên hệ",
        "phone_label": "Số điện thoại di động",
        "contact_btn": "Gửi Thông Tin Liên Hệ",
        "contact_success": "✅ Đã ghi nhận thông tin! Chuyên viên thẩm định sẽ gọi hỗ trợ trong vòng 24 giờ làm việc.",
        "contact_err": "Vui lòng nhập chính xác Email hoặc Số điện thoại.",
        "rejected_title": "❌ RẤT TIẾC! HỒ SƠ CHƯA ĐẠT TIÊU CHUẨN (REJECTED)",
        "rejected_msg": "Điểm tín dụng hiện tại (**{score} điểm**) chưa đạt ngưỡng an toàn tối thiểu theo quy định cấp tín dụng.",
        "mascot_speech": "> *Chân thành xin lỗi quý khách! VietBank rất tiếc chưa thể hỗ trợ khoản vay lần này. Quý khách vui lòng duy trì lịch sử trả nợ tốt hoặc cập nhật thêm thu nhập và đăng ký lại sau 3 tháng nhé!*",
        # Explainability
        "explain_header": "💡 Giải Thích Ý Nghĩa Chỉ Số Đánh Giá",
        "exp_score_title": "**1. Điểm Tín Dụng (Credit Score):**",
        "exp_score_desc": "- **Thang điểm:** Từ 300 đến 850 điểm.\n- **Ý nghĩa:** Chỉ số tổng hợp phản ánh mức độ uy tín tài chính. Điểm càng cao, năng lực trả nợ càng tốt.\n- **Phương pháp:** Quy đổi theo công thức chuẩn hóa Scorecard ngân hàng dựa trên trọng số rủi ro ($WOE \\times \\beta$).",
        "exp_pd_title": "**2. Xác Suất Vỡ Nợ (Probability of Default - PD):**",
        "exp_pd_desc": "- **Mức PD dự báo:** **{pd:.2f}%**\n- **Ý nghĩa:** Tỷ lệ rủi ro dự kiến khách hàng phát sinh nợ quá hạn quá 90 ngày trong 12 tháng tới.\n- **Ứng dụng:** Là cơ sở để phân hạng rủi ro và trích lập dự phòng theo chuẩn Basel II/III.",
        "rule_expander": "🔍 Bảng Quy Định Phân Hạng Rủi Ro & Hành Động (Risk Matrix)",
        "tbl_col1": "Khung Điểm (Score)",
        "tbl_col2": "Phân Hạng Rủi Ro",
        "tbl_col3": "Quyết Định Thẩm Định",
        "tbl_col4": "Quy Trình Xử Lý",
    },
    "EN": {
        "title": "🏦 Automated Credit Risk Assessment & Scoring System",
        "subtitle": "Automated credit application intake and analysis portal fully compliant with banking risk standards.",
        "sidebar_title": "📋 Credit Application Form",
        "engine_label": "Active Model Run:",
        "sec_personal": "👤 I. Personal & Financial Profile",
        "sec_loan": "💳 II. Loan Request & Credit Details",
        # Field Labels & Helps
        "age": "Applicant Age",
        "age_help": "Valid age range between 18 and 100 years old.",
        "income": "Annual Gross Income ($)",
        "income_help": "Total yearly pre-tax income from salaried employment and verified sources.",
        "emp_length": "Employment Tenure (years)",
        "emp_length_help": "Total accumulated length of employment at the current organization.",
        "loan_amnt": "Requested Loan Amount ($)",
        "loan_amnt_help": "The total credit limit requested by the applicant.",
        "loan_int_rate": "Expected Interest Rate (%/yr)",
        "loan_int_rate_help": "Applicable annual interest rate for the requested loan package.",
        "cred_hist_len": "Credit History Length (years)",
        "cred_hist_len_help": "Years since the applicant opened their first credit account or loan.",
        "home_ownership": "Home Ownership Status",
        "home_ownership_help": "Residential legal status of the applicant.",
        "loan_intent": "Purpose of Loan",
        "loan_intent_help": "Primary financial intent or sector for fund disbursement.",
        "loan_grade": "Loan Risk Grade",
        "loan_grade_help": "Risk rating from A (Lowest Risk) to G (Highest Risk).",
        "default_on_file": "Historical Default Record",
        "default_on_file_help": "Whether the applicant has a past delinquency or default record on credit bureau.",
        # Buttons & Messages
        "confirm_check": "I hereby certify that all provided disclosures are accurate and true to the best of my knowledge.",
        "submit_btn": "🚀 Submit Application & Assess",
        "guide_info": "👈 Please complete the application form in the Sidebar and click **'Submit Application & Assess'** to begin.",
        "warn_confirm": "⚠️ Please mark the confirmation checkbox before submitting your application.",
        "spinner_msg": "🔄 AI engine is evaluating application parameters and calculating default probability... Please wait.",
        "res_header": "📊 Comprehensive Assessment Results",
        "kpi_score": "Credit Score",
        "kpi_pd": "Probability of Default (PD)",
        "kpi_ratio": "Loan-to-Income Ratio",
        "dec_header": "🏆 Credit Underwriting Decision",
        # Approval Scenarios
        "approved_title": "✅ AUTOMATICALLY APPROVED",
        "approved_msg": "Congratulations! Your credit score reached **{score} points**. Your loan application of **${amount:,.0f}** has been automatically approved.",
        "pending_title": "⚠️ MANUAL REVIEW REQUIRED (PENDING REVIEW)",
        "pending_msg": "Your credit score stands at **{score} points**. To complete disbursement, Risk Management requires verifying additional documentation.",
        "contact_form_title": "**Please leave your contact details for a Credit Officer to follow up directly:**",
        "email_label": "Contact Email Address",
        "phone_label": "Mobile Phone Number",
        "contact_btn": "Submit Contact Details",
        "contact_success": "✅ Information recorded! A credit specialist will contact you within 24 business hours.",
        "contact_err": "Please enter a valid Email address or Phone number.",
        "rejected_title": "❌ APPLICATION REJECTED",
        "rejected_msg": "Your current credit score (**{score} points**) does not meet the bank's minimum safety threshold for credit extension.",
        "mascot_speech": "> *Sincere apologies! VietBank regrets that we cannot proceed with this loan request. Please maintain a healthy credit history or update your income and re-apply in 3 months!*",
        # Explainability
        "explain_header": "💡 Detailed Metric Interpretations",
        "exp_score_title": "**1. Credit Score:**",
        "exp_score_desc": "- **Scale:** Ranging from 300 to 850 points.\n- **Meaning:** Composite indicator of financial trustworthiness. Higher scores signify stronger repayment capacity.\n- **Methodology:** Converted using standard banking Scorecard mathematical scaling ($WOE \\times \\beta$).",
        "exp_pd_title": "**2. Probability of Default (PD):**",
        "exp_pd_desc": "- **Predicted PD:** **{pd:.2f}%**\n- **Meaning:** Expected probability of 90+ days delinquency over the next 12 months.\n- **Application:** Serves as the core foundation for risk grading and Basel II/III provisioning.",
        "rule_expander": "🔍 View Risk Matrix Rules & System Actions",
        "tbl_col1": "Score Range",
        "tbl_col2": "Risk Grade",
        "tbl_col3": "Underwriting Decision",
        "tbl_col4": "System Process",
    },
}


# 3. AUTOMATIC PIPELINE LOADING FROM LATEST RUN
def get_latest_run_dir(base_dir="artifacts/runs"):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory {base_dir} does not exist.")
    subdirs = glob.glob(f"{base_dir}/*/")
    if not subdirs:
        raise FileNotFoundError("No model run artifacts found.")
    return max(subdirs, key=os.path.getmtime)


@st.cache_resource
def load_pipelines():
    base_dir = "demo_artifacts"

    cleaner = joblib.load(os.path.join(base_dir, "cleaner.pkl"))
    woe_tf = joblib.load(os.path.join(base_dir, "woe_transformer.pkl"))
    model = joblib.load(os.path.join(base_dir, "baseline_logistic_model.pkl"))
    scaler = joblib.load(os.path.join(base_dir, "score_scaler.pkl"))
    return cleaner, woe_tf, model, scaler, base_dir


try:
    cleaner, woe_transformer, model_trainer, score_scaler, active_run_dir = (
        load_pipelines()
    )
except Exception as e:
    st.error(
        f"⚠️ System initialization error: {e}\nPlease re-run main.py to create model artifacts."
    )
    st.stop()

# 4. SIDEBAR PANEL: LANGUAGE SWITCHER, LOGO & PROFESSIONAL FORM
with st.sidebar:
    # 4.1 LANGUAGE SWITCHER
    lang_choice = st.radio(
        "🌐 Language / Ngôn ngữ", ["🇻🇳 Tiếng Việt", "🇬🇧 English"], horizontal=True
    )
    lang = "VN" if "Tiếng Việt" in lang_choice else "EN"
    txt = I18N[lang]

    st.markdown("---")
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=70)
    st.title("VIETBANK CREDIT RISK")
    st.caption(
        f"{txt['engine_label']} {os.path.basename(os.path.normpath(active_run_dir))}"
    )
    st.markdown("---")

    st.subheader(txt["sidebar_title"])

    # 4.2 PROFESSIONAL INPUT FORM WITH TOOLTIPS
    with st.form("customer_input_form"):
        st.markdown(txt["sec_personal"])
        age = st.number_input(
            txt["age"], min_value=18, max_value=100, value=28, help=txt["age_help"]
        )
        income = st.number_input(
            txt["income"], min_value=0, value=65000, step=1000, help=txt["income_help"]
        )
        emp_length = st.number_input(
            txt["emp_length"],
            min_value=0.0,
            max_value=50.0,
            value=4.0,
            step=0.5,
            help=txt["emp_length_help"],
        )
        home_ownership = st.selectbox(
            txt["home_ownership"],
            ["RENT", "MORTGAGE", "OWN", "OTHER"],
            help=txt["home_ownership_help"],
        )

        st.markdown(txt["sec_loan"])
        loan_amnt = st.number_input(
            txt["loan_amnt"],
            min_value=500,
            value=10000,
            step=500,
            help=txt["loan_amnt_help"],
        )
        loan_int_rate = st.number_input(
            txt["loan_int_rate"],
            min_value=0.0,
            max_value=40.0,
            value=11.0,
            step=0.5,
            help=txt["loan_int_rate_help"],
        )
        loan_intent = st.selectbox(
            txt["loan_intent"],
            [
                "PERSONAL",
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION",
            ],
            help=txt["loan_intent_help"],
        )
        loan_grade = st.selectbox(
            txt["loan_grade"],
            ["A", "B", "C", "D", "E", "F", "G"],
            help=txt["loan_grade_help"],
        )
        cred_hist_length = st.number_input(
            txt["cred_hist_len"],
            min_value=0,
            max_value=50,
            value=5,
            help=txt["cred_hist_len_help"],
        )
        default_on_file = st.selectbox(
            txt["default_on_file"], ["N", "Y"], help=txt["default_on_file_help"]
        )

        st.markdown("---")
        confirm_checkbox = st.checkbox(txt["confirm_check"])
        submit_button = st.form_submit_button(
            label=txt["submit_btn"], use_container_width=True
        )

# 5. MAIN INTERFACE PANEL
st.title(txt["title"])
st.markdown(txt["subtitle"])

if not submit_button:
    st.info(txt["guide_info"])
else:
    if not confirm_checkbox:
        st.warning(txt["warn_confirm"])
    else:
        # 5.1 LOADING SPINNER EFFECT
        with st.spinner(txt["spinner_msg"]):
            time.sleep(1.2)  # Simulate AI inference latency

            # Compute loan-to-income ratio
            loan_percent_income = round(loan_amnt / income, 2) if income > 0 else 0.0

            # Construct COMPLETE Input DataFrame matching model expectations
            input_df = pd.DataFrame(
                [
                    {
                        "person_age": age,
                        "person_income": income,
                        "person_emp_length": emp_length,
                        "loan_amnt": loan_amnt,
                        "loan_int_rate": loan_int_rate,
                        "loan_percent_income": loan_percent_income,
                        "cb_person_cred_hist_length": cred_hist_length,
                        "person_home_ownership": home_ownership,
                        "loan_intent": loan_intent,
                        "loan_grade": loan_grade,
                        "cb_person_default_on_file": default_on_file,
                    }
                ]
            )

            # Pipeline Transformations
            X_clean = cleaner.transform(input_df)
            X_woe = woe_transformer.transform(X_clean)

            from src.utils import extract_structural_bins

            X_bins = extract_structural_bins(X_clean, woe_transformer)

            pd_prob = model_trainer.predict_probability(X_woe)[0]
            credit_score = int(score_scaler.transform(X_bins).iloc[0])

        # 5.2 RENDER METRICS & DECISIONS
        st.markdown("---")
        st.subheader(txt["res_header"])

        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        col_kpi1.metric(txt["kpi_score"], f"{credit_score} pts")
        col_kpi2.metric(txt["kpi_pd"], f"{pd_prob * 100:.2f}%")
        col_kpi3.metric(txt["kpi_ratio"], f"{loan_percent_income * 100:.1f}%")

        st.markdown(f"### {txt['dec_header']}")

        # 5.3 THREE-TIER APPROVAL SCENARIOS
        if credit_score >= 650:
            st.balloons()
            st.success(f"### {txt['approved_title']}")
            st.write(txt["approved_msg"].format(score=credit_score, amount=loan_amnt))

        elif 580 <= credit_score < 650:
            st.warning(f"### {txt['pending_title']}")
            st.write(txt["pending_msg"].format(score=credit_score))

            with st.form("contact_form"):
                st.markdown(txt["contact_form_title"])
                c_col1, c_col2 = st.columns(2)
                user_email = c_col1.text_input(
                    txt["email_label"], placeholder="name@example.com"
                )
                user_phone = c_col2.text_input(
                    txt["phone_label"], placeholder="+84 901 234 567"
                )

                submit_contact = st.form_submit_button(txt["contact_btn"])
                if submit_contact:
                    if user_email or user_phone:
                        st.success(txt["contact_success"])
                    else:
                        st.error(txt["contact_err"])

        else:
            st.error(f"### {txt['rejected_title']}")
            st.write(txt["rejected_msg"].format(score=credit_score))

            mascot_col1, mascot_col2 = st.columns([1, 3])
            with mascot_col1:
                st.image(
                    "https://cdn-icons-png.flaticon.com/512/4076/4076549.png", width=120
                )
            with mascot_col2:
                st.markdown(txt["mascot_speech"])

        # 5.4 EXPLAINABILITY SECTION
        st.markdown("---")
        st.subheader(txt["explain_header"])

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown(txt["exp_score_title"])
            st.write(txt["exp_score_desc"])

        with exp_col2:
            st.markdown(txt["exp_pd_title"])
            st.write(txt["exp_pd_desc"].format(pd=pd_prob * 100))

        # Risk Matrix Lookup Rules Table
        with st.expander(txt["rule_expander"]):
            if lang == "VN":
                st.table(
                    pd.DataFrame(
                        {
                            txt["tbl_col1"]: [
                                "Score >= 650",
                                "580 <= Score < 650",
                                "Score < 580",
                            ],
                            txt["tbl_col2"]: [
                                "Thấp (Low Risk)",
                                "Trung bình (Medium Risk)",
                                "Cao (High Risk)",
                            ],
                            txt["tbl_col3"]: [
                                "✅ Phê duyệt ngay",
                                "⚠️ Yêu cầu thẩm định bổ sung",
                                "❌ Từ chối hồ sơ",
                            ],
                            txt["tbl_col4"]: [
                                "Tự động cấp hạn mức tối đa",
                                "Gửi thông tin cho Chuyên viên Tín dụng",
                                "Hiển thị linh vật xin lỗi",
                            ],
                        }
                    )
                )
            else:
                st.table(
                    pd.DataFrame(
                        {
                            txt["tbl_col1"]: [
                                "Score >= 650",
                                "580 <= Score < 650",
                                "Score < 580",
                            ],
                            txt["tbl_col2"]: ["Low Risk", "Medium Risk", "High Risk"],
                            txt["tbl_col3"]: [
                                "✅ Instant Approval",
                                "⚠️ Manual Review Required",
                                "❌ Application Rejected",
                            ],
                            txt["tbl_col4"]: [
                                "Issue Maximum Limit",
                                "Collect Contact Details for Credit Officer",
                                "Display Apology Mascot",
                            ],
                        }
                    )
                )
