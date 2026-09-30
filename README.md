# 🏦 Credit Risk Modeling 

An end-to-end machine learning project that predicts the **probability of loan default** for Lauki Finance customers, converts it into a **credit score (300–900)**, and assigns a **risk rating**. Deployed as a live **Streamlit web app**.

## 🚀 Live Demo
🔗 **https://hanadismail-credit-risk-modeling-appmain-m9szfx.streamlit.app/**

## 📌 Objective
Build a model the Risk Unit can use to measure credit risk, meeting these success criteria:
- AUC, Gini > 85
- KS statistic > 40
- Max KS in the first 3 deciles
- Interpretable model

## 📊 Data
Two years of loan data from Lauki Finance (Feb 2022–Feb 2024 for training/validation, Mar–May 2024 held out as out-of-time test data), across three sources in `dataset/`:

| File | Key fields |
|------|-----------|
| **customers.csv** | age, gender, marital status, employment status, income, dependents, residence type, address tenure, city/state/zipcode |
| **loans.csv** | loan purpose, loan type, sanction amount, loan amount, net disbursement, tenure, principal outstanding, bank balance at application, default flag |
| **bureau_data.csv** | open/closed accounts, total loan months, delinquent months, total DPD, enquiry count, credit utilization ratio |

## ⚙️ Modeling Pipeline
1. **Merge** customers + loans + bureau data; target = `default`
2. **Clean**: fix invalid `loan_purpose` values (replaced with mode)
3. **Feature engineer**: `loan_to_income`, `deliquency_ratio`, `avg_dpd_per_deliquency`
4. **Select features**: IV, VIF, and domain knowledge
5. **Preprocess**: min-max scaling on numeric features, one-hot encoding on categoricals
6. **Split**: 75% train / 25% test
7. **Train**: Logistic Regression, XGBoost, Random Forest
8. **Tune**: RandomizedSearchCV, Optuna
9. **Evaluate**: AUC, Gini, KS statistic, classification report

## 🧠 Top Predictive Variables (by Information Value)
| Variable | IV | Insight |
|---|---|---|
| `credit_utilization_ratio` | 2.35 | Higher credit usage sharply increases default risk |
| `delinquency_ratio` | 0.71 | More delinquent months strongly linked to default |
| `loan_to_income` | 0.47 | Higher loan-to-income raises default likelihood |
| `avg_dpd_per_delinquency` | 0.40 | More days past due per delinquency raises risk |
| `loan_purpose` | 0.36 | Certain purposes carry more default risk |
| `residence_type` | 0.24 | Moderate effect on default risk |
| `loan_tenure_months` | 0.21 | Longer tenure increases default risk |
| `loan_type` | 0.16 | Minor influence on default risk |
| `age` | 0.08 | Minimal effect |
| `number_of_open_accounts` | 0.08 | More open accounts, slightly higher risk |

## 📈 Model Performance
| Model | AUC | Gini |
|---|---|---|
| Logistic Regression | 98 | 96 |
| **XGBoost** | **99** | **96** |
| Random Forest | 97 | 95 |

**Final selected model (deployed): Logistic Regression** — chosen for interpretability, with performance essentially matching XGBoost.

**Overall model evaluation:** AUC 98%, Gini 96%, Top-3-decile capture rate 99.53% — comfortably clears the success criteria above.

## 🏷️ Credit Score Ratings
| Score | Rating |
|---|---|
| 300–499 | Poor |
| 500–649 | Average |
| 650–749 | Good |
| 750–900 | Excellent |

## 📂 Project Structure
```
Credit-Risk-Modeling/
├── app/
│   ├── artifacts/
│   │   └── model_data.joblib        # Model weights, scaler, and feature schemas
│   ├── main.py                      # Streamlit web application frontend
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

## 🖥️ App Inputs / Outputs
**Inputs:** age, income, loan amount, loan tenure, average DPD, delinquency ratio, credit utilization ratio, open accounts, residence type, loan purpose, loan type.

**Outputs:** default probability, credit score, rating.

## 🛠️ Tech Stack
Python · Pandas · NumPy · Scikit-learn · XGBoost · Optuna · Statsmodels · Streamlit

## 🚀 Running Locally
```bash
git clone https://github.com/HanadIsmail/Credit-Risk-Modeling.git
cd Credit-Risk-Modeling
pip install -r requirements.txt
streamlit run app/main.py
```

## ⚠️ Notes
- Built for a learning/portfolio project modeled on a Codebasics ML course case study (Lauki Finance).
- Not used for real lending decisions.

