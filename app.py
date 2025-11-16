# app.py (unchanged from your last corrected version)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from src.preprocessing import load_ibm, add_market_midpoint, synthesize_recruitment
from src.metrics import *
from src.rules_engine import generate_recommendations
from src.ml_model import prepare_features # This is crucial and correctly imported

st.set_page_config(layout="wide", page_title="HR Decision Support Dashboard")

# Load data
DATA_PATH = "data/ibm_hr_attrition.csv"
MID_PATH = "data/market_midpoints.csv"
RECRUIT_PATH = "data/recruitment_events.csv"

df = load_ibm(DATA_PATH)
df = add_market_midpoint(df, MID_PATH)

# Synthesize recruitment if missing
if not os.path.exists(RECRUIT_PATH):
    synthesize_recruitment(RECRUIT_PATH, n=120)
recruit_df = pd.read_csv(RECRUIT_PATH)

# Sidebar filters
st.sidebar.header("Filters")
dept_filter = st.sidebar.multiselect("Department", options=df['Department'].unique(), default=list(df['Department'].unique()))
df_f = df[df['Department'].isin(dept_filter)]

# Sidebar simulation
st.sidebar.header("Scenario Simulation")
attrition_reduction_pct = st.sidebar.slider("Reduce attrition by (%)", 0, 50, 0)
training_increase_pct = st.sidebar.slider("Increase training coverage by (%)", 0, 50, 0)

# --- Base Metrics ---
metrics = {}
metrics['turnover_rate'] = turnover_rate(df_f)
metrics['retention_rate'] = retention_rate(df_f)
metrics['stability_index'] = stability_index(df_f)
metrics['avg_years'] = avg_years_of_stay(df_f)
metrics['labour_cost_per_fte'] = labour_cost_per_fte(df_f)
metrics['labour_cost_total'] = labour_cost_total(df_f)
metrics['CompaRatio'] = compa_ratio_avg(df_f)
metrics['training_coverage'] = training_coverage(df_f)
metrics['avg_performance'] = avg_performance_rating(df_f)

# Recruitment KPIs
metrics['time_to_fill'] = time_to_fill(recruit_df)
metrics['cost_per_hire'] = cost_per_hire(recruit_df)
metrics['offer_acceptance_rate'] = offer_acceptance_rate(recruit_df)

# --- Apply Scenario Adjustments ---
sim_metrics = metrics.copy()
if attrition_reduction_pct > 0:
    sim_metrics['turnover_rate'] *= (1 - attrition_reduction_pct/100)
if training_increase_pct > 0:
    sim_metrics['training_coverage'] *= (1 + training_increase_pct/100)

# --- Dashboard Header ---
st.title("HR Decision Support Dashboard")

# KPI Cards
k1, k2, k3, k4 = st.columns(4)
k1.metric("Turnover Rate", f"{sim_metrics['turnover_rate']:.2f}%", f"{metrics['turnover_rate'] - sim_metrics['turnover_rate']:.2f}%")
k2.metric("Retention Rate", f"{sim_metrics['retention_rate']:.2f}%")
k3.metric("Avg Compa Ratio", f"{sim_metrics['CompaRatio']:.2f}")
k4.metric("Training Coverage", f"{sim_metrics['training_coverage']:.2f}%")

k5, k6, k7 = st.columns(3)
k5.metric("Time to Fill (days)", f"{sim_metrics['time_to_fill']:.1f}")
k6.metric("Cost per Hire", f"{sim_metrics['cost_per_hire']:.0f}")
k7.metric("Offer Acceptance Rate", f"{sim_metrics['offer_acceptance_rate']:.2f}%")

# Charts (dynamic with simulation)
st.subheader("Attrition by Department (Simulated)")
adj_attr = df_f.copy()
# Ensure 'AttritionFlag' exists for this line. If 'Attrition' is raw, you might need to map it.
# Assuming load_ibm or other steps create 'AttritionFlag'.
if 'AttritionFlag' in adj_attr.columns:
    adj_attr['AttritionFlag'] = adj_attr['AttritionFlag'] * (1 - attrition_reduction_pct/100)
    # Only show if there are positive attrition values after adjustment
    if not adj_attr[adj_attr['AttritionFlag'] > 0]['Department'].empty:
        st.bar_chart(adj_attr[adj_attr['AttritionFlag'] > 0]['Department'].value_counts())
    else:
        st.info("No simulated attrition to display after reduction.")
else:
    st.warning("AttritionFlag column not found for 'Attrition by Department' chart.")


st.subheader("Compa Ratio Distribution")
fig, ax = plt.subplots()
ax.hist(df_f['CompaRatio'].dropna(), bins=20)
ax.set_xlabel("Compa Ratio")
ax.set_ylabel("Count")
st.pyplot(fig)

# Recommendations
st.subheader("Automated Recommendations")
recs = generate_recommendations({
    'turnover_rate': sim_metrics['turnover_rate'],
    'CompaRatio': sim_metrics['CompaRatio'],
    'training_coverage': sim_metrics['training_coverage'],
    'avg_performance': sim_metrics['avg_performance'],
})
for r in recs:
    st.write("- ", r)

# Scenario Savings
st.subheader("Scenario Cost Savings")
if attrition_reduction_pct > 0:
    base_cost = total_cost_of_turnover(df_f, avg_cost_replacement=45000)
    new_est_left = int((sim_metrics['turnover_rate']/100) * headcount(df_f))
    new_cost = new_est_left * 45000
    savings = base_cost - new_cost
    st.success(f"Estimated savings in turnover cost: ₹{savings:,.0f}")

# ML Model: Attrition Prediction
st.subheader("Attrition Risk (ML)")
model_path = "outputs/rf_attrition.joblib"
if os.path.exists(model_path):
    clf = joblib.load(model_path)

    # Use the same prepare_features function that was used during training
    X_for_prediction = prepare_features(df) # <--- THIS IS CORRECT FOR PATH A

    print("Model features:", clf.feature_names_in_)
    print("Input features:", list(X_for_prediction.columns))

    # The additional checks are good, but if Path A is followed,
    # these should ideally pass without issue.
    if len(clf.feature_names_in_) != len(X_for_prediction.columns):
        st.error(f"Feature count mismatch! Model expects {len(clf.feature_names_in_)} features, but got {len(X_for_prediction.columns)}.")
        st.error(f"Model features: {list(clf.feature_names_in_)}")
        st.error(f"Input features: {list(X_for_prediction.columns)}")
    else:
        if not all(clf.feature_names_in_ == X_for_prediction.columns):
            st.warning("Feature names or their order mismatch between model and input, but count is same. Attempting prediction...")
            X_for_prediction = X_for_prediction[list(clf.feature_names_in_)]


    df['attrition_proba'] = clf.predict_proba(X_for_prediction)[:,1]
    st.write("Top 10 high-risk employees (probability-adjusted)")
    st.dataframe(df.sort_values('attrition_proba', ascending=False)[['EmployeeNumber','JobRole','Department','attrition_proba']].head(10))
else:
    st.info("⚠️ Train the model first: run `src/ml_model.py` (or notebooks/trainml.py, if applicable)")

# Survey Upload
st.subheader("Import Survey Results (CSV)")
survey_file = st.file_uploader("Upload survey CSV", type=['csv'])
if survey_file:
    s_df = pd.read_csv(survey_file)
    st.write("Survey sample:")
    st.dataframe(s_df.head())
    for col in ['JobSatisfaction_Score','TrainingEffectiveness_Score','CompensationFairness_Score']:
        if col in s_df.columns:
            st.write(f"Average {col}: {s_df[col].mean():.2f}")