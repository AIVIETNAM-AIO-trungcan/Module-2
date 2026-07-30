# 💳 Credit Risk Scoring & Decision Support System

> **Module 02 Project - End-to-End MLOps Credit Scorecard Pipeline & Interactive Web Application**

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/trungcan94/AIO_moddule_2_model)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-28A745?style=flat)]()

---

## 📌 1. Project Overview

The **Credit Risk Scoring and Decision Support System** is a comprehensive, production-grade MLOps solution engineered for Commercial Banking and FinTech underwriting automation.

Built on strict credit risk principles (WOE Encoding, Logistic Regression Scaling, PDO Calibration), the system generates highly interpretable Business Scorecards while adhering to rigorous regulatory compliance standards:

- **Strict Anti-Data Leakage Architecture:** Enforces early 3-way Stratified Partitioning (Train / Validation / Test) prior to any data cleaner fitting or WOE binning.
- **Banking-Grade Outlier Mitigation:** Applies Winsorization (Capping) to preserve 100% of applicant records while maintaining dedicated binning categories for missing/NaN values.
- **Dual-Branch Feature Selection:** Compares a Baseline VIF + IV screening branch against a Challenger LASSO 10-Fold Cross-Validation (1-SE Rule) feature subset.
- **Financial Calibration & Regulatory Compliance:** Achieves a Mean Absolute Error (MAE) of $\approx 0.39$ score points, near-perfect calibration variance ($1.0017 \approx 1.0$), and enforces strictly negative risk coefficients ($\beta < 0$).
- **MLOps Cloud Auto-Sync:** Packages production binaries into `model.zip` upon training completion and automatically syncs them with Hugging Face Hub for instant Streamlit Cloud deployment.
- **Role-Based Security Control:** Integrates an Underwriter Security Gate (`admin123` passcode) to hide sensitive score attribution charts and proprietary scaling tables from public applicant views.

---

## 👥 2. Team & Contributors

- **Project Title:** Credit Risk Scoring and Decision Support System
- **Team Name:** `Porsche_Club`
- **Organization:** AIO CONQUER
- **Repository:** [AIVIETNAM-AIO-trungcan/Module-2](https://github.com/AIVIETNAM-AIO-trungcan/Module-2)

| No. | Member Name          | Role & Core Responsibilities                                                                                                                                        | GitHub Profile                                                           |
| :-: | :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------- |
|  1  | **Trần Phương Bình** | **Tech Leader**<br>• System architecture oversight & technical compliance<br>• Risk modeling audit & project roadmap execution                                      | [@AIVIETNAM-AIO-TPBINH](https://github.com/AIVIETNAM-AIO-TPBINH)         |
|  2  | **Đào Trung Cần**    | **AI Engineer (Pipeline)**<br>• Master pipeline engineering (`main.py`, `src/`)<br>• Scorecard scaling, PDO calibration & Hugging Face MLOps auto-sync              | [@AIVIETNAM-AIO-trungcan](https://github.com/AIVIETNAM-AIO-trungcan)     |
|  3  | **Cao Bá Hoàng**     | **AI Engineer (Model)**<br>• Model selection, hyperparameter tuning & LASSO CV optimization<br>• Regulatory beta audits & risk coefficient verification             | [@AIVIETNAM-AIO-CaoBaHoang](https://github.com/AIVIETNAM-AIO-CaoBaHoang) |
|  4  | **Nguyễn Tùng**      | **AI Engineer (Data)**<br>• Tier-1 Data Cleaner, outlier capping & Tier-2 WOE Binning engine<br>• Data leakage prevention & feature information value (IV) analysis | [@AIVIETNAM-AIO-Jayn79](https://github.com/AIVIETNAM-AIO-Jayn79)         |
|  5  | **Vũ Khánh Vy**      | **QA / Reviewer**<br>• Underwriting security testing & Streamlit UI quality assurance<br>• Documentation audit & business scorecard rulebook validation             | [@Whoami-404-pip](https://github.com/Whoami-404-pip)                     |

---

## 📂 3. Repository Directory Structure

```text
Module-2/
├── .vscode/                         # IDE workspace configurations
├── artifacts/                       # MLOps isolated run storage
│   └── runs/                        # Dynamic execution run logs (e.g., 2026-07-30_run_11)
│       ├── data/                    # Processed WOE datasets and score-transformed tables
│       ├── metrics/                 # JSON reports of model performance (KS, Gini, MAE)
│       ├── models/                  # Serialized binary model packages (.pkl files)
│       ├── plots/                   # Generated evaluation charts (ROC, KS Curves, Monotonicity)
│       └── tables/                  # Business Scorecard Lookup Tables & QA audit logs (.csv)
├── asset/                           # Project visual assets & mascot images
├── checklist/                       # Governance & project verification checklists
├── data/                            # Dataset directory (contains raw credit data)
├── demo_artifacts/                  # Fixed sample artifacts for demo fallback
├── notebook/                        # Exploratory Data Analysis (EDA) & experimental notebooks
├── src/                             # Core Library Modules
│   ├── __init__.py                  # Python package initializer
│   ├── config.py                    # System path constants & configurations
│   ├── data_loader.py               # Raw data loader & Stratified Train/Val/Test partitioning
│   ├── feature_selection.py         # Tier-3 Dual-Branch Feature Selector (VIF, IV, LASSO)
│   ├── inference.py                 # Production Inference Engine with Auto-Cloud Recovery
│   ├── model.py                     # Logistic Regression Model Trainer & Regulatory Beta Audits
│   ├── preprocessing.py             # Tier-1 Statistical Cleaner (Capping) & Tier-2 WOE Engine
│   ├── scorecard.py                 # Financial Score Scaling (Base Score, PDO, Factor/Offset)
│   └── utils.py                     # Performance Evaluation, ROC/KS Plots & Audit Export Helpers
├── .env                             # Environment secrets file (HF_TOKEN) - Excluded via .gitignore
├── .gitignore                       # Git rules ignoring binaries, cache files, and zip packages
├── app.py                           # Application UI Entrypoint (Streamlit Interactive Web App)
├── config.yaml                      # Centralized pipeline configuration & hyperparameters
├── LICENSE                          # Open-source license declaration
├── main.py                          # Master Pipeline Execution Engine (Training, Audit & Cloud Sync)
├── README.md                        # Technical project documentation
├── requirements.txt                 # Project environment dependencies
└── ui_config.yaml                   # UI layout settings, i18n localization, and admin passcode
```

## 🔄 4. Project Pipeline Architecture

The end-to-end workflow from raw data ingestion to automated cloud deployment:.

```mermaid
flowchart TD
    A[Raw Credit Applicant Data] --> B[Step 1-3: Data Ingestion &<br>Stratified Split Isolation]
    B --> TrainSet[Training Set 64%]
    B --> ValSet[Validation Set 16%]
    B --> TestSet[Testing Set 20%]

    TrainSet --> C[Step 4: Tier-1 Statistical Cleaner<br>Outlier Capping]
    C --> D[Step 5: Tier-2 Dynamic<br>WOE Binning & Encoding]

    D --> E[Step 6: Tier-3 Dual-Branch<br>Feature Selection]
    E --> F1[Baseline Branch:<br>VIF & IV Filtering]
    E --> F2[Challenger Branch:<br>LASSO 10-Fold CV]

    F1 --> G[Step 7: Dual Model Training &<br>Regulatory Beta Verification]
    F2 --> G

    G --> H[Champion Model Auto-Selection<br>via Validation KS/Gini Metrics]

    H --> I[Step 8: Scorecard Scaling &<br>Financial PDO Calibration Audit]
    I --> J[MLOps Artifact Packaging<br>model.zip Creation]

    J --> K{HF_TOKEN Detected?}
    K -- Yes --> L[Auto-Upload model.zip to<br>Hugging Face Hub Dataset]
    K -- No --> M[Local Artifact Storage Lock]

    L --> N[Production Web App<br>Streamlit Dashboard app.py]
    M --> N
```

## 📊 5. Technical Performance & Audit Metrics

The selected Champion Model (`SUBSET`) exhibits excellent discrimination capability without signs of overfitting across all populations:

| Metric Category             | Performance Indicator                                                                   | Benchmark / Target                  | Audit Status  |
| :-------------------------- | :-------------------------------------------------------------------------------------- | :---------------------------------- | :-----------: |
| **Model Discrimination**    | **KS Index:** `0.6407` (Test Set)<br>**Gini Coefficient:** `0.7550`                     | KS > 0.40<br>Gini > 0.60            | 🟢 Excellent  |
| **Classification Accuracy** | **Overall Accuracy:** `87.68%`                                                          | Accuracy > 80%                      |  🟢 Optimal   |
| **Score Calibration**       | **Mean Absolute Error (MAE):** `0.3922` pts<br>**Calibration Variance Ratio:** `1.0017` | MAE < 1.0 pt<br>Ratio $\approx 1.0$ | 🟢 Calibrated |
| **Regulatory Trend Audit**  | **Economic Logic:** $100\%$ negative coefficients ($\beta < 0$)                         | Monotonic Trend                     | 🟢 Compliant  |

> > 📄 **Full Technical Documentation:** [Download Full Technical Audit Report (.PDF)](asset/Tech_Report.pdf)

---

## 🌐 6. Product Demo & Media Showcase

### 🎥 Streamlit Application Live Preview

![Credit Risk Scoring App Demo](asset/demo_app.gif)

---

> 🚀 **Live Interactive Web Application**
>
> Experience the automated credit scoring and decision support system online:
> 👉 **[CLICK HERE FOR LIVE STREAMLIT DEMO](https://module-2-porsche-club.streamlit.app/)**

> 🎬 **Full Walkthrough demo**
> Demo for tab 1: Single Application

<p align="center">
  <img src="asset/demo_tab1.gif" alt="Streamlit Credit Risk Scorecard Demo" width="100%" style="border-radius: 8px;"/>
  <br>
  <em>Interactive Web Application Walkthrough: Single Application.</em>
</p>

> Demo for tab 2: Batch CSV Scoring

<p align="center">
  <img src="asset/demo_tab2.gif" alt="Streamlit Credit Risk Scorecard Demo" width="100%" style="border-radius: 8px;"/>
  <br>
  <em>Interactive Web Application Walkthrough: Batch CSV Scoring.</em>
</p>
---

## 🛠️ 7. Quick Start Guide (Local Setup)

### Prerequisites

- **Python 3.10+** (Python 3.13 recommended)
- **Git** installed on local environment

### Step-by-Step Installation

1. **Clone the repository:**

```Bash
git clone https://github.com/AIVIETNAM-AIO-trungcan/Module-2.git
cd Module-2
```

2. **Install environment dependencies:**

```Bash
pip install -r requirements.txt
```

3. **Configure environment variables (Optional for Cloud Sync):**
   Create a .env file in the root directory to enable automatic model upload to Hugging Face:

```env
HF_TOKEN=hf_your_actual_write_access_token
```

4. **Run the master training pipeline:**

```Bash
python main.py
```

5. **Launch the Streamlit Web Application:**

```Bash
streamlit run app.py
```

---

_Developed by **Porsche_Club** - Module 02 Credit Risk Scoring & Decision Support System._
