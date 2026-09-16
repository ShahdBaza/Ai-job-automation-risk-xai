import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

st.set_page_config(
    page_title="AI Job Automation Risk & Explainability (XAI)",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Job Automation Risk Analyzer")
st.caption(
    "Automated Risk Assessment & AI Decision Reasoning (Powered by SHAP &"
    " Random Forest)"
)

feature_cols = [
    'annual_mean_salary_log',
    'total_employment_log',
    'manual_dexterity',
    'creativity_level',
    'critical_thinking',
    'major_13',
    'major_15',
    'major_17',
    'major_19',
    'major_21',
    'major_23',
    'major_25',
    'major_27',
    'major_29',
    'major_31',
    'major_33',
    'major_35',
    'major_37',
    'major_39',
    'major_41',
    'major_43',
    'major_45',
    'major_47',
    'major_49',
    'major_51',
    'major_53',
]

feature_names_en = {
    'annual_mean_salary_log': 'Annual Salary Level',
    'total_employment_log': 'Workforce Size (Employment)',
    'manual_dexterity': 'Manual Dexterity',
    'creativity_level': 'Creativity & Originality',
    'critical_thinking': 'Critical Thinking & Problem Solving',
}


@st.cache_resource
def load_all():
  # مسارات نسبية لتعمل محلياً وعلى Streamlit Cloud مباشرة
  model_path = "real_model.pkl"
  csv_path = "predicted_jobs.csv"

  if not os.path.exists(model_path) or not os.path.exists(csv_path):
    st.error(
        "Files not found! Please ensure 'real_model.pkl' and"
        " 'predicted_jobs.csv' are in the same folder as app.py."
    )
    st.stop()

  model = joblib.load(model_path)
  explainer = shap.TreeExplainer(model)
  df = pd.read_csv(csv_path)

  job_col = None
  for col in [
      'Job Title',
      'title',
      'job_title',
      'occupation',
      'Occupation',
      'Title',
  ]:
    if col in df.columns:
      job_col = col
      break
  if job_col is None:
    job_col = df.columns[0]

  return model, explainer, df, job_col


model, explainer, df_jobs, job_col = load_all()

# Sidebar: الاختيار
st.sidebar.header("🎯 Occupation Selection")
job_options = sorted(df_jobs[job_col].dropna().unique().tolist())
selected_job = st.sidebar.selectbox("Choose Job Title:", job_options)



if selected_job:
  job_data = df_jobs[df_jobs[job_col] == selected_job].iloc[0]

  features_available = [c for c in feature_cols if c in df_jobs.columns]
  x_input = pd.DataFrame([job_data[features_available]])

  risk_pct = float(job_data['risk_pct'])
  risk_level = str(job_data['risk_level'])

  # حسابات SHAP
  shap_values = explainer(x_input)
  sv = shap_values[0].values
  f_names = x_input.columns.tolist()

  shap_df = pd.DataFrame({'feature': f_names, 'shap_val': sv})
  shap_df['feature_readable'] = shap_df['feature'].map(
      lambda x: feature_names_en.get(x, x)
  )

  risk_drivers = shap_df[shap_df['shap_val'] > 0.01].sort_values(
      by='shap_val', ascending=False
  )
  protective_factors = shap_df[shap_df['shap_val'] < -0.01].sort_values(
      by='shap_val', ascending=True
  )

  # عرض النتيجة الرئيسية
  st.subheader(f"📌 Assessment for: {selected_job}")

  col1, col2, col3 = st.columns([1.2, 1, 1])

  with col1:
    st.markdown(f"### **Automation Risk:** `{risk_pct}%`")
    if 'High' in risk_level:
      st.error(f"### Level: {risk_level}")
    elif 'Medium' in risk_level:
      st.warning(f"### Level: {risk_level}")
    else:
      st.success(f"### Level: {risk_level}")
    st.progress(int(min(max(risk_pct, 0), 100)))

  with col2:
    salary_val = job_data.get('annual_mean_salary', None)
    if salary_val is not None and not pd.isna(salary_val):
      st.metric(label="Annual Salary", value=f"${salary_val:,.0f}")
    crit_val = job_data.get('critical_thinking', None)
    if crit_val is not None and not pd.isna(crit_val):
      st.metric(label="Critical Thinking", value=f"{crit_val}/5.5")

  with col3:
    creat_val = job_data.get('creativity_level', None)
    if creat_val is not None and not pd.isna(creat_val):
      st.metric(label="Creativity Level", value=f"{creat_val}/5.5")
    man_val = job_data.get('manual_dexterity', None)
    if man_val is not None and not pd.isna(man_val):
      st.metric(label="Manual Dexterity", value=f"{man_val}/5.5")

  st.markdown("---")

  # الأسباب والعوامل المؤثرة (Explainability)
  st.subheader("💡 Key Drivers & Reasons (Why this score?)")
  cause_col1, cause_col2 = st.columns(2)

  with cause_col1:
    st.markdown("**Risk-Increasing Factors (Pushed Risk Up):**")
    if not risk_drivers.empty:
      for _, row in risk_drivers.head(3).iterrows():
        st.write(
            f"🔺 **{row['feature_readable']}**: Increases susceptibility to"
            f" automation (Impact: +{row['shap_val']:.2f})"
        )
    else:
      st.write(
          "No significant attributes driving automation risk; naturally"
          " resilient."
      )

  with cause_col2:
    st.markdown("**Protective Factors (Kept Risk Down):**")
    if not protective_factors.empty:
      for _, row in protective_factors.head(3).iterrows():
        st.write(
            f"🛡️ **{row['feature_readable']}**: Shields occupation by favoring"
            f" human expertise (Impact: {row['shap_val']:.2f})"
        )
    else:
      st.write(
          "Current tasks exhibit limited human complexity to buffer against"
          " automation."
      )

  st.markdown("---")

  # رسم الـ Waterfall Plot لكل وظيفة
  st.subheader("📊 SHAP Waterfall Plot")
  fig, ax = plt.subplots(figsize=(9, 4))
  shap.plots.waterfall(shap_values[0], max_display=7, show=False)
  st.pyplot(fig)
  plt.close()