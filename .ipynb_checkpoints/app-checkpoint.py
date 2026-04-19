# ============================================================
# Customer Churn Prediction System
# Author: Puneeth
# ============================================================
# Feature encoding:
#   Gender  --> 1 = Female, 0 = Male
#   Churn   --> 1 = Yes,    0 = No
# Feature order for model: ['Age', 'Gender', 'Tenure', 'MonthlyCharges']
# ============================================================
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1e293b; }
    /* Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-card .label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        color: #0d9488;
    }
    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .risk-high   { background: #450a0a; color: #fca5a5; border: 1px solid #ef4444; }
    .risk-medium { background: #431407; color: #fdba74; border: 1px solid #f97316; }
    .risk-low    { background: #052e16; color: #86efac; border: 1px solid #22c55e; }
    /* Section headers */
    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e293b;
    }
    /* Result box */
    .result-box {
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        text-align: center;
    }
    .result-churn    { background: #1c0505; border: 2px solid #ef4444; }
    .result-no-churn { background: #021205; border: 2px solid #22c55e; }
    .result-title { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
    .result-sub   { font-size: 14px; color: #94a3b8; }
    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔮 Single Prediction", "📂 Bulk Prediction"])

# ── Load model & scaler ──────────────────────────────────────
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    model  = joblib.load("model.pkl")
    return scaler, model
scaler, model = load_artifacts()
# ── Sidebar — Model Info ─────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Model Information")
    st.markdown("---")
    st.markdown('<div class="metric-card"><div class="label">Algorithm</div><div class="value" style="font-size:18px">Random Forest</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><div class="label">Estimators</div><div class="value">128</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><div class="label">Max Depth</div><div class="value">5</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><div class="label">Class Weighting</div><div class="value" style="font-size:18px">Balanced</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔑 Feature Importance")
    features   = ["Tenure", "Monthly Charges", "Age", "Gender"]
    importance = [65.8, 29.9, 4.3, 0.0]
    colors     = ["#0d9488", "#14b8a6", "#5eead4", "#ccfbf1"]
    fig_imp = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v}%" for v in importance],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
    ))
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=40, t=0, b=0),
        height=180,
        xaxis=dict(visible=False),
        yaxis=dict(tickfont=dict(color="#94a3b8", size=11)),
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown("---")
    st.caption("📌 Tenure & Monthly Charges are the strongest predictors of churn.")
# ── Main Content ─────────────────────────────────────────────
with tab1:
    
    st.markdown("# 📊 Customer Churn Prediction")
    st.markdown("Enter customer details below to predict churn likelihood.")
    st.markdown("---")
    col_form, col_results = st.columns([1, 1.2], gap="large")
# ── Left: Input Form ─────────────────────────────────────────
with col_form:
    st.markdown('<p class="section-header">Customer Details</p>', unsafe_allow_html=True)
    age = st.slider("Age", min_value=18, max_value=90, value=35,
                    help="Customer's age in years")
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12,
                       help="How long the customer has been with the company")
    monthly_charge = st.slider("Monthly Charges ($)", min_value=30, max_value=150, value=65,
                               help="The customer's current monthly bill")
    st.markdown("---")
    # Summary preview
    st.markdown('<p class="section-header">Input Summary</p>', unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        "Field":  ["Age", "Gender", "Tenure", "Monthly Charges"],
        "Value":  [str(age), gender, f"{tenure} months", f"${monthly_charge}"],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.markdown("")
    predict_button = st.button("🔍 Predict Churn", use_container_width=True, type="primary")
# ── Right: Results ───────────────────────────────────────────
with col_results:
    st.markdown('<p class="section-header">Prediction Results</p>', unsafe_allow_html=True)
    if predict_button:
        # Prepare input
        gender_encoded = 1 if gender == "Female" else 0
        x = np.array([[age, gender_encoded, tenure, monthly_charge]])
        x_scaled = scaler.transform(x)
        prediction  = model.predict(x_scaled)[0]
        probability = model.predict_proba(x_scaled)[0][1]
        stay_prob   = 1 - probability
        # ── Risk level ──────────────────────────────────────
        if probability >= 0.7:
            risk_level = "High Risk"
            risk_class = "risk-high"
            risk_icon  = "🔴"
        elif probability >= 0.4:
            risk_level = "Medium Risk"
            risk_class = "risk-medium"
            risk_icon  = "🟠"
        else:
            risk_level = "Low Risk"
            risk_class = "risk-low"
            risk_icon  = "🟢"
        # ── Result banner ────────────────────────────────────
        if prediction == 1:
            st.balloons()
            st.markdown(f"""
            <div class="result-box result-churn">
                <div class="result-title" style="color:#f87171">⚠️ Likely to Churn</div>
                <div class="result-sub">Churn probability: <strong style="color:#f87171">{probability*100:.1f}%</strong></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-no-churn">
                <div class="result-title" style="color:#4ade80">✅ Not Likely to Churn</div>
                <div class="result-sub">Retention probability: <strong style="color:#4ade80">{stay_prob*100:.1f}%</strong></div>
            </div>""", unsafe_allow_html=True)
        # Risk badge
        st.markdown(f'<p style="text-align:center;margin:8px 0 16px">'
                    f'<span class="risk-badge {risk_class}">{risk_icon} {risk_level}</span>'
                    f'</p>', unsafe_allow_html=True)
        # ── Probability gauge chart ──────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(probability * 100, 1),
            number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 36}},
            title={"text": "Churn Probability", "font": {"color": "#94a3b8", "size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#64748b"}},
                "bar":  {"color": "#ef4444" if probability > 0.5 else "#22c55e"},
                "bgcolor": "#1e293b",
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0,  40],  "color": "#052e16"},
                    {"range": [40, 70],  "color": "#431407"},
                    {"range": [70, 100], "color": "#450a0a"},
                ],
                "threshold": {
                    "line": {"color": "#f59e0b", "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30, b=10, l=20, r=20),
            height=220,
            font={"color": "#e2e8f0"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        # ── Probability bar chart ────────────────────────────
        fig_bar = go.Figure(go.Bar(
            x=["Will Churn", "Will Stay"],
            y=[probability, stay_prob],
            marker_color=["#ef4444", "#22c55e"],
            text=[f"{probability*100:.1f}%", f"{stay_prob*100:.1f}%"],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=13),
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            height=220,
            yaxis=dict(visible=False, range=[0, 1.25]),
            xaxis=dict(tickfont=dict(color="#94a3b8", size=12)),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        # ── Churn type & recommended action ─────────────────
        if prediction == 1:
            st.markdown("---")
            st.markdown('<p class="section-header">Diagnosis & Recommended Action</p>', unsafe_allow_html=True)
            if tenure < 12:
                churn_type = "🆕 New Customer Churn"
                solution   = "Offer onboarding support, a welcome discount, or assign a dedicated account manager to improve early-stage experience."
            elif monthly_charge > 80:
                churn_type = "💸 High Cost Churn"
                solution   = "Propose a discounted plan, loyalty pricing, or a bundle upgrade that provides better perceived value."
            else:
                churn_type = "😞 Service Dissatisfaction Churn"
                solution   = "Trigger a proactive check-in, resolve open support tickets, and offer a satisfaction survey with incentive."
            st.warning(f"**Churn Type:** {churn_type}")
            st.info(f"**💡 Suggested Action:** {solution}")
    else:
        # Placeholder state
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color: #475569;">
            <div style="font-size: 48px; margin-bottom: 16px;">🎯</div>
            <div style="font-size: 18px; font-weight: 600; color: #64748b;">Ready to predict</div>
            <div style="font-size: 14px; margin-top: 8px;">Fill in the customer details on the left and click <strong>Predict Churn</strong>.</div>
        </div>
        """, unsafe_allow_html=True)

# ── Bulk Prediction Section ─────────────────────────────
with tab2:
    
    st.markdown("---")
    st.markdown("## 📂 Bulk Prediction (Upload CSV)")
    
    file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if file:
        df = pd.read_csv(file)
    
        st.write("📊 Preview Data", df.head())
    
        try:
            # Encode Gender
            df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    
            X = df[["Age", "Gender", "Tenure", "MonthlyCharges"]]
            X_scaled = scaler.transform(X)
    
            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]
    
            df["Prediction"] = preds
            df["Churn Probability"] = probs
    
            st.write("✅ Prediction Results", df.head())
    
            csv = df.to_csv(index=False).encode("utf-8")
    
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
    
        except Exception as e:
            st.error("❌ Error processing file. Make sure CSV has columns: Age, Gender, Tenure, MonthlyCharges")
        
# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:13px;'>"
    "Built by <strong style='color:#94a3b8'>Puneeth</strong> · Machine Learning University Project · 2026"
    "</p>",
    unsafe_allow_html=True,)