import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Load Model ----------------
@st.cache_resource
def load_pipeline():
    model_path = Path(__file__).resolve().parent.parent / "models" / "credit_risk_pipeline.pkl"
    return joblib.load(model_path)

pipeline = load_pipeline()

# ---------------- Session State Init ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Feature Importance Data ----------------
FEATURE_NAMES = [
    "person_age", "person_income", "person_emp_length", "person_home_ownership",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"
]

FEATURE_EXPLANATIONS = {
    "loan_percent_income": "Loan amount relative to your annual income",
    "loan_int_rate": "Interest rate on the loan",
    "loan_grade": "Creditworthiness grade assigned to the loan",
    "person_income": "Your annual income",
    "loan_amnt": "Total loan amount requested",
    "cb_person_cred_hist_length": "Length of your credit history",
    "person_emp_length": "Years of employment",
    "cb_person_default_on_file": "Previous loan default on record",
    "person_age": "Applicant age",
    "person_home_ownership": "Home ownership status",
    "loan_intent": "Purpose of the loan"
}

# ---------------- Header ----------------
st.title("💳 Credit Risk Assessment")
st.caption(
    "AI-powered system to evaluate loan default risk and support transparent credit decisions."
)

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("📋 Applicant Information")

# Personal Information
with st.sidebar.expander("👤 Personal Details", expanded=True):
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Annual Income ($)", min_value=1000, value=50000, step=1000)
    person_emp_length = st.number_input("Employment Length (Years)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    person_home_ownership = st.selectbox(
        "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )

# Loan Details
with st.sidebar.expander("💰 Loan Details", expanded=True):
    loan_intent = st.selectbox(
        "Loan Purpose",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        format_func=lambda x: x.replace("HOMEIMPROVEMENT", "Home Improvement").replace("DEBTCONSOLIDATION", "Debt Consolidation").title()
    )
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, value=10000, step=500)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.5)
    loan_percent_income = st.slider("Loan as % of Income", 0.0, 1.0, 0.2, 0.01)

# Credit History
with st.sidebar.expander("📊 Credit History", expanded=True):
    cb_person_default_on_file = st.selectbox("Previous Default on Record", ["N", "Y"], format_func=lambda x: "Yes" if x == "Y" else "No")
    cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, max_value=50, value=5)

st.sidebar.divider()
predict_button = st.sidebar.button("🔍 Assess Credit Risk", type="primary")

# ---------------- Main Content Tabs ----------------
tab_results, tab_explain, tab_governance, tab_history = st.tabs([
    "📊 Results", "🔍 Explanation", "🏛️ Model Info", "📜 History"
])

# ---------------- Prediction Logic ----------------
if predict_button:
    input_df = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
    }])

    prob_default = pipeline.predict_proba(input_df)[0][1]
    risk_score = int(100 - prob_default * 100)

    if risk_score >= 80:
        risk_level, decision, color = "Low Risk", "Approved", "🟢"
    elif risk_score >= 60:
        risk_level, decision, color = "Medium Risk", "Review Required", "🟡"
    else:
        risk_level, decision, color = "High Risk", "Rejected", "🔴"

    # Store in session state
    st.session_state.last_prediction = {
        "prob_default": prob_default,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "decision": decision,
        "color": color,
        "input_data": input_df.iloc[0].to_dict()
    }

    # Add to history
    st.session_state.history.append({
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Score": risk_score,
        "Risk Level": risk_level,
        "Decision": decision,
        "Loan Amount": f"${loan_amnt:,}",
        "Income": f"${person_income:,}"
    })

# ---------------- Results Tab ----------------
with tab_results:
    if "last_prediction" in st.session_state:
        pred = st.session_state.last_prediction
        
        st.subheader("Assessment Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Default Probability", f"{pred['prob_default']:.1%}")
        with col2:
            st.metric("Credit Risk Score", f"{pred['risk_score']}/100")
        with col3:
            st.metric("Recommendation", f"{pred['color']} {pred['decision']}")

        st.divider()

        # Visual Risk Gauge
        st.subheader("Risk Score Gauge")
        score = pred['risk_score']
        st.progress(score / 100)
        
        col_gauge1, col_gauge2, col_gauge3 = st.columns(3)
        col_gauge1.caption("🔴 High Risk (0-59)")
        col_gauge2.caption("🟡 Medium Risk (60-79)")
        col_gauge3.caption("🟢 Low Risk (80-100)")

        # Risk Category Display
        if pred['risk_level'] == "Low Risk":
            st.success(f"✅ **{pred['risk_level']}**: This applicant shows strong creditworthiness indicators.")
        elif pred['risk_level'] == "Medium Risk":
            st.warning(f"⚠️ **{pred['risk_level']}**: This application requires additional review before approval.")
        else:
            st.error(f"❌ **{pred['risk_level']}**: This applicant shows elevated default risk indicators.")

    else:
        st.info("👈 Enter applicant details in the sidebar and click **Assess Credit Risk** to see results.")

# ---------------- Explanation Tab ----------------
with tab_explain:
    if "last_prediction" in st.session_state:
        st.subheader("Understanding the Assessment")
        
        st.markdown("""
        The credit risk score is calculated by analyzing multiple factors about the applicant 
        and their loan request. Here are the key factors that typically influence the assessment:
        """)

        # Get feature importances from the model
        try:
            # Access the XGBoost model from the pipeline
            model = pipeline.named_steps.get('classifier') or pipeline.named_steps.get('model') or pipeline[-1]
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Get preprocessor output feature names if available
                try:
                    preprocessor = pipeline.named_steps.get('preprocessor') or pipeline[0]
                    feature_names = preprocessor.get_feature_names_out()
                except:
                    feature_names = FEATURE_NAMES
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_names[:len(importances)],
                    'Importance': importances
                }).sort_values('Importance', ascending=False)

                # Top 3 Features
                st.markdown("### 🎯 Top 3 Contributing Factors")
                
                top_features = importance_df.head(3)
                for i, (_, row) in enumerate(top_features.iterrows(), 1):
                    feature = row['Feature']
                    # Clean feature name for display
                    display_name = feature.split('__')[-1] if '__' in feature else feature
                    display_name = display_name.replace('_', ' ').title()
                    
                    # Get explanation
                    base_feature = feature.split('__')[0] if '__' in feature else feature
                    explanation = FEATURE_EXPLANATIONS.get(base_feature, "This factor influences the risk assessment")
                    
                    st.markdown(f"**{i}. {display_name}**")
                    st.caption(explanation)
                    st.progress(float(row['Importance']))
                    st.write("")

                # Feature Importance Chart (in expander)
                with st.expander("📊 Full Feature Importance Chart"):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    top_10 = importance_df.head(10)
                    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_10)))[::-1]
                    
                    bars = ax.barh(range(len(top_10)), top_10['Importance'].values, color=colors)
                    ax.set_yticks(range(len(top_10)))
                    
                    # Clean labels
                    labels = [f.split('__')[-1].replace('_', ' ').title() for f in top_10['Feature']]
                    ax.set_yticklabels(labels)
                    
                    ax.set_xlabel('Importance Score')
                    ax.set_title('Top 10 Features by Importance')
                    ax.invert_yaxis()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            else:
                st.info("Feature importance data is not available for this model type.")
                
        except Exception as e:
            st.warning("Unable to extract feature importances from the model.")
            
        # SHAP section (optional, behind expander)
        with st.expander("🔬 Advanced: SHAP Analysis"):
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** provides detailed per-prediction explanations.
            
            For performance reasons, SHAP analysis is not computed by default. 
            To enable SHAP visualizations, you can:
            1. Run the explainability notebook locally
            2. Generate SHAP values for specific predictions
            
            This ensures the app remains fast and responsive on Streamlit Cloud.
            """)
    else:
        st.info("👈 Make a prediction first to see the explanation.")

# ---------------- Governance Tab ----------------
with tab_governance:
    st.subheader("🏛️ Model Information & Governance")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("### Model Details")
        st.markdown("""
        | Property | Value |
        |----------|-------|
        | **Model Type** | XGBoost Classifier |
        | **Pipeline** | Unified sklearn Pipeline |
        | **Evaluation Metric** | ROC-AUC |
        | **Training Data** | Kaggle Credit Risk Dataset |
        """)
        
        st.markdown("""
        **Dataset Source**: [Credit Risk Dataset on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
        """)

    with col_info2:
        st.markdown("### Intended Use")
        st.markdown("""
        This model is designed for:
        - **Educational purposes** and portfolio demonstration
        - **Risk assessment screening** to support human decision-making
        - **Explainable AI** research and development
        
        This model is **NOT** intended for:
        - Automated credit decisions without human review
        - Production lending systems
        - Regulatory compliance decisions
        """)

    st.divider()
    
    st.markdown("### ⚠️ Limitations & Ethical Considerations")
    st.warning("""
    **Important Disclaimer**: This is a demonstration model and should NOT be used for actual credit decisions.
    
    - **Bias Risk**: The training data may contain historical biases that could perpetuate unfair outcomes
    - **Context Limitations**: The model cannot account for individual circumstances not captured in the features
    - **Regulatory Compliance**: This model has not been validated for regulatory compliance (e.g., ECOA, FCRA)
    - **Human Oversight Required**: All predictions should be reviewed by qualified professionals
    """)

# ---------------- History Tab ----------------
with tab_history:
    st.subheader("📜 Prediction History")
    st.caption("Recent assessments from this session (stored in memory only)")
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, hide_index=True)
        
        col_clear, col_count = st.columns([1, 3])
        with col_clear:
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                if "last_prediction" in st.session_state:
                    del st.session_state.last_prediction
                st.rerun()
        with col_count:
            st.caption(f"Total predictions: {len(st.session_state.history)}")
    else:
        st.info("No predictions yet. Make an assessment to see history.")

# ---------------- Footer ----------------
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #888; padding: 10px;">
        Built by <strong>Aditya Negi</strong> | 
        <a href="https://www.linkedin.com/in/Adityanegi748" target="_blank">LinkedIn</a> | 
        <a href="https://github.com/adityanegiuk99" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
