# 🏦 Credit Risk Modeling & Scorecard System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning and Credit Scorecard solution that evaluates loan applicants, estimates the **Probability of Default (PD)**, and computes a calibrated **Credit Score (300 – 900)** with credit ratings. The system is deployed as an interactive, production-ready web application using Streamlit.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Datasets](#-datasets)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Credit Scorecard Methodology](#-credit-scorecard-methodology)
- [Streamlit Web Application](#-streamlit-web-application)
- [Local Setup & Installation](#-local-setup--installation)
- [Deploying to Streamlit Cloud](#-deploying-to-streamlit-cloud)
- [Tech Stack](#-tech-stack)

---

## 🚀 Project Overview

Credit risk assessment is a critical component of retail banking and fintech underwriting. This repository provides:
1. **End-to-End Analysis & Modeling** in Jupyter Notebook (`Credit_Risk_Modeling.ipynb`), exploring demographic, loan, and credit bureau data.
2. **Feature Engineering & Transformation**: Weight of Evidence (WoE), Information Value (IV), Outlier handling, and MinMax scaling.
3. **Model Benchmarking & Hyperparameter Tuning**: Logistic Regression, Random Forest, and XGBoost optimized with Optuna and RandomizedSearchCV.
4. **Credit Scorecard Calibration**: Mapping default probability into an industry-standard 300 to 900 credit score with risk tiers.
5. **Interactive Web App**: A clean, responsive Streamlit dashboard (`app/main.py`) for underwriting decisions.

---

## 📂 Repository Structure

```text
Credit-Risk-Modeling/
├── app/
│   ├── artifacts/
│   │   └── model_data.joblib        # Model weights, scaler, and feature schemas
│   ├── main.py                     # Streamlit web application frontend
│   └── prediction_helper.py         # Data preprocessing and inference pipeline
├── artifacts/
│   └── model_data.joblib            # Root model artifact backup
├── dataset/
│   ├── bureau_data.csv              # Credit bureau history & repayment records
│   ├── customers.csv                # Applicant demographic & financial profile
│   └── loans.csv                    # Loan application & disbursement details
├── Credit_Risk_Modeling.ipynb       # Complete EDA, modeling & evaluation notebook
├── requirements.txt                 # Project dependencies for local & cloud runs
├── .gitignore                       # Git ignore configuration
└── README.md                        # Project documentation
```

---

## 📊 Datasets

The model leverages three integrated datasets located in the [`dataset/`](dataset/) directory:

| Dataset | Records & Description | Key Features |
| :--- | :--- | :--- |
| **`customers.csv`** | Demographic and income details | `age`, `number_of_dependants`, `income`, `years_at_current_address`, `bank_balance_at_application`, `residence_type` |
| **`loans.csv`** | Loan contract details | `loan_amount`, `loan_tenure_months`, `sanction_amount`, `processing_fee`, `loan_purpose`, `loan_type` |
| **`bureau_data.csv`** | Historical credit bureau bureau performance | `delinquency_ratio`, `avg_dpd_per_delinquency`, `credit_utilization_ratio`, `number_of_open_accounts`, `enquiry_count` |

---

## 🧠 Machine Learning Pipeline

1. **Exploratory Data Analysis (EDA)**: Distribution analysis, correlation matrices, and delinquency pattern inspection.
2. **Data Cleaning & Preprocessing**:
   - Outlier capping and treatment.
   - Missing value imputation.
   - Categorical one-hot encoding (`residence_type`, `loan_purpose`, `loan_type`).
3. **Feature Engineering**:
   - **Loan-to-Income (LTI) Ratio**: $\frac{\text{Loan Amount}}{\text{Income}}$
   - **Credit Utilization Ratio**: Percentage of revolving credit currently drawn.
   - **Delinquency Metrics**: Proportion of delinquent past loans and Average Days Past Due (Avg DPD).
4. **Feature Scaling**: `MinMaxScaler` applied across numerical features to ensure balanced weight contribution.
5. **Model Benchmarking**:
   - **Logistic Regression**: High interpretability, aligned with Basel II scorecard requirements.
   - **Random Forest**: Non-linear ensemble comparison.
   - **XGBoost**: Gradient boosting with hyperparameter optimization via **Optuna** and **RandomizedSearchCV**.
6. **Model Validation**:
   - ROC-AUC Curve
   - Kolmogorov-Smirnov (KS) Statistic
   - Gini Coefficient
   - Decile Rank-Ordering Analysis

---

## 📈 Credit Scorecard Methodology

The final production model employs a calibrated Logistic Regression scorecard to compute log-odds and default probabilities:

$$\text{Log-Odds } (x) = \beta_0 + \sum_{i=1}^{n} \beta_i X_i$$

$$\text{Probability of Default (PD)} = \frac{1}{1 + e^{-x}}$$

$$\text{Non-Default Probability} = 1 - \text{PD}$$

### Scorecard Formula
$$\text{Credit Score} = \text{Base Score} + (\text{Non-Default Probability} \times \text{Scale Length})$$
- **Base Score**: `300`
- **Scale Length**: `600` (yielding a range of **300 to 900**)

### Risk Rating Tiers

| Credit Score Range | Rating Category | Default Risk | Underwriting Advisory |
| :---: | :---: | :---: | :--- |
| **750 – 900** | 🟢 **Excellent** | Very Low | Fast-track approval with prime interest rates |
| **650 – 749** | 🔵 **Good** | Low | Standard approval under regular policy |
| **500 – 649** | 🟡 **Average** | Medium | Conditional approval with additional verification or collateral |
| **300 – 499** | 🔴 **Poor** | High | High default probability; recommend decline |

---

## 💻 Streamlit Web Application

The interactive web UI allows credit underwriters and analysts to input applicant details and immediately view:
- **Default Probability**: Percentage likelihood of default.
- **Credit Score**: Score between 300 and 900 with a visual progress bar indicator.
- **Credit Rating**: Tiered badge with automated underwriting advice.

### Inputs Required:
- **Demographics & Financials**: Age, Annual Income, Loan Amount Requested, Loan Tenure (months).
- **Credit Bureau Metrics**: Avg DPD, Delinquency Ratio (%), Credit Utilization Ratio (%), Open Loan Accounts.
- **Loan Specifics**: Residence Type (`Owned`, `Rented`, `Mortgage`), Loan Purpose (`Education`, `Home`, `Auto`, `Personal`), Loan Type (`Unsecured`, `Secured`).

---

## 🛠️ Local Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/HanadIsmail/Credit-Risk-Modeling.git
cd Credit-Risk-Modeling
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app/main.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Cloud

You can deploy this application directly to **[Streamlit Community Cloud](https://share.streamlit.io/)** in minutes:

1. **Fork or Push** this repository to your GitHub account (`HanadIsmail/Credit-Risk-Modeling`).
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **"New app"**.
4. Configure the deployment settings:
   - **Repository**: `HanadIsmail/Credit-Risk-Modeling`
   - **Branch**: `main`
   - **Main file path**: `app/main.py`
5. Click **"Deploy!"**. Streamlit Cloud will install all dependencies from `requirements.txt` and launch your live application with a public URL!

---

## 🧰 Tech Stack

- **Core & Runtime**: Python 3.10+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost, Optuna, SciPy
- **Model Serialization**: Joblib
- **Web App**: Streamlit
- **Visualization**: Matplotlib, Seaborn

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
