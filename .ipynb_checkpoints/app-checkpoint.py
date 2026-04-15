# Gender --> 1 Female   0 Male 
# Churn --> 1 Yes   0 No
# Scaler is exported as scaler.pkl
#Model is exported as model.pkl
#Order of x -> ['Age', 'Gender', 'Tenure', 'MonthlyCharges'] 

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model & scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.set_page_config(page_title="Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction System ")
st.markdown("### churn analytics dashboard 🚀")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📂 Bulk Prediction", "📊 Insights"])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("📝 Single Customer Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 10, 100, 30)
        tenure = st.number_input("Tenure", 0, 100, 10)

    with col2:
        monthlycharge = st.number_input("Monthly Charges", 30, 150)
        gender = st.selectbox("Gender", ["Male", "Female"])

    if st.button("🚀 Predict"):
        gender_selected = 1 if gender == "Female" else 0

        x = np.array([[age, gender_selected, tenure, monthlycharge]])
        x_scaled = scaler.transform(x)

        prediction = model.predict(x_scaled)[0]
        probability = model.predict_proba(x_scaled)[0][1]
        churn_percent = round(probability * 100, 2)

        # Risk Level
        if churn_percent < 30:
            risk = "🟢 Low"
        elif churn_percent < 70:
            risk = "🟡 Medium"
        else:
            risk = "🔴 High"

        if prediction == 1:
            st.error(f"⚠️ Likely to churn ({churn_percent}%)")
        else:
            st.success(f"✅ Not likely to churn ({churn_percent}%)")

        st.info(f"Risk Level: {risk}")

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("📂 Upload CSV for Bulk Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.write("📊 Preview Data", df.head())

        # Preprocess
        df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

        X = df[["Age", "Gender", "Tenure", "MonthlyCharges"]]
        X_scaled = scaler.transform(X)

        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]

        df["Prediction"] = preds
        df["Churn Probability"] = probs

        st.write("✅ Results", df.head())

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("📊 Insights Dashboard")

    st.markdown("### Sample Insights (from dataset)")

    sample_data = pd.DataFrame({
        "Category": ["Low Risk", "Medium Risk", "High Risk"],
        "Count": [120, 80, 60]
    })

    st.bar_chart(sample_data.set_index("Category"))

    st.markdown("### 📈 Key Observations")
    st.write("- Customers with low tenure are more likely to churn")
    st.write("- High monthly charges increase churn risk")
    st.write("- Retention strategies should target new users")

st.markdown("---")
