# 🤖 AI Job Automation Risk & Explainability (XAI)

An end-to-end Machine Learning and Explainable AI (XAI) system designed to estimate occupational automation susceptibility. The system couples robust regression models with **SHAP (SHapley Additive exPlanations)** to unpack model reasoning into interpretable, human-centric insights.

---

## 📌 Project Overview
* **Domain:** Labor Economics, Artificial Intelligence & Predictive Modeling
* **Core Objective:** Predict the likelihood of occupational automation and diagnose key protective vs. vulnerable task attributes.
* **Explainability:** Employs SHAP TreeExplainer to deliver local (job-specific waterfall plots) and global feature attribution.
* **Deployment:** Interactive web dashboard built with Streamlit.

---

## 🏗️ System Architecture & Workflow

1. **Data Engineering & Integration:**
   * Merged labor and occupational benchmark sets including state-level automation data, Bureau of Labor Statistics (BLS) salary and employment metrics, and O*NET abilities/skills databases.
   * Extracted targeted human skill metrics: *Creativity Level (Originality)*, *Manual Dexterity*, *Critical Thinking*, *Complex Problem Solving*, and *Social Perceptiveness*.

2. **Feature Engineering & Preprocessing:**
   * Log transformations (`log1p`) applied to high-skew continuous variables (`annual_mean_salary`, `total_employment`).
   * Categorical major occupational codes extracted via SOC two-digit grouping and transformed using One-Hot Encoding (`drop_first=True`).

3. **Benchmarking & Validation:**
   * Multi-model evaluation across Linear Regression, KNN, Decision Trees, SVR (Pipeline with StandardScaler), Gradient Boosting, and Random Forest.
   * Selection of tuned **Random Forest Regressor** (`n_estimators=150`, `max_depth=12`, `random_state=42`) as the production model based on superior generalization and stable $R^2$ scores.

4. **Interpretability & XAI:**
   * Global feature analysis showing the shielding influence of higher creativity and critical thinking against automation.
   * Dynamic local inference isolating positive risk drivers from protective attributes per occupation.

---

## 📊 Directory Structure

```text
├── app.py                     # Streamlit web application & visualization dashboard
├── Final_project_Ai_risk.ipynb    # Research notebook (Data prep, EDA, training, SHAP analysis)
├── real_model.pkl             # Serialized production Random Forest model
├── final_clean_ai_risk.csv         # Processed dataset with calculated risk scores & categories
├── requirements.txt           # Environment dependencies
└── README.md                  # Project documentation
