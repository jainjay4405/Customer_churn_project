import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ChurnAI",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("customer_churn_model.pkl")

model = load_model()

# =========================================================
# CUSTOM CSS ONLY
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

h1, h2, h3 {
    color: white !important;
}

.stMarkdown p {
    color: #cbd5e1;
}

div[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #263244;
    padding: 20px;
    border-radius: 15px;
}

div[data-testid="stMetricValue"] {
    color: #60a5fa;
}

.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    font-size: 16px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚡ ChurnAI")

    st.caption("Customer Intelligence Platform")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🔮 Churn Prediction",
            "📊 About Project"
        ]
    )

    st.divider()

    st.caption(
        "Machine Learning Project\n\n"
        "Customer Churn Prediction"
    )


# =========================================================
# PREDICTION PAGE
# =========================================================

if page == "🔮 Churn Prediction":

    st.title("📊 Customer Churn Predictor")

    st.write(
        "Predict whether a telecom customer is likely to churn."
    )

    st.divider()

    # -----------------------------------------------------
    # CUSTOMER PROFILE
    # -----------------------------------------------------

    st.subheader("👤 Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

    with col2:

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    with col3:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )


    # -----------------------------------------------------
    # SERVICES
    # -----------------------------------------------------

    st.subheader("🛠️ Customer Services")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col2:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col3:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col4:

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col2:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    # -----------------------------------------------------
    # BILLING
    # -----------------------------------------------------

    st.subheader("💰 Billing Information")

    col1, col2 = st.columns(2)

    with col1:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

    with col2:

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1000.0,
            step=10.0
        )


    st.divider()


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    predict = st.button(
        "🔮 Predict Customer Churn"
    )


    if predict:

        # Feature engineering

        if tenure > 0:
            average_monthly_spend = total_charges / tenure
        else:
            average_monthly_spend = monthly_charges


        # Input dataframe

        input_data = pd.DataFrame({

            "gender": [gender],

            "SeniorCitizen": [senior_citizen],

            "Partner": [partner],

            "Dependents": [dependents],

            "tenure": [tenure],

            "PhoneService": [phone_service],

            "MultipleLines": [multiple_lines],

            "InternetService": [internet_service],

            "OnlineSecurity": [online_security],

            "OnlineBackup": [online_backup],

            "DeviceProtection": [device_protection],

            "TechSupport": [tech_support],

            "StreamingTV": [streaming_tv],

            "StreamingMovies": [streaming_movies],

            "Contract": [contract],

            "PaperlessBilling": [paperless_billing],

            "PaymentMethod": [payment_method],

            "MonthlyCharges": [monthly_charges],

            "TotalCharges": [total_charges],

            "AverageMonthlySpend": [
                average_monthly_spend
            ]

        })


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        probability_percentage = probability * 100


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.divider()

        st.subheader("📈 Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Churn Probability",
                f"{probability_percentage:.2f}%"
            )

        with col2:

            st.metric(
                "Tenure",
                f"{tenure} months"
            )

        with col3:

            st.metric(
                "Monthly Charges",
                f"${monthly_charges:.2f}"
            )


        st.progress(
            min(probability, 1.0)
        )


        if prediction == 1:

            st.error(
                "⚠️ Customer is predicted to churn"
            )

        else:

            st.success(
                "✅ Customer is predicted to stay"
            )


        # Risk level

        if probability_percentage >= 70:

            st.error("🔴 Risk Level: HIGH")

        elif probability_percentage >= 40:

            st.warning("🟠 Risk Level: MEDIUM")

        else:

            st.success("🟢 Risk Level: LOW")


# =========================================================
# ABOUT PAGE
# =========================================================

else:

    st.title("📊 About ChurnAI")

    st.write(
        "An end-to-end Machine Learning project "
        "for predicting telecom customer churn."
    )

    st.divider()

    st.subheader("🎯 Project Objective")

    st.write(
        "The goal is to predict whether a customer "
        "is likely to leave the telecom service."
    )

    st.subheader("🧠 Machine Learning Pipeline")

    st.write("""
    1. Data Cleaning
    2. Exploratory Data Analysis
    3. Feature Engineering
    4. Encoding
    5. Feature Scaling
    6. Model Training
    7. Model Evaluation
    8. Hyperparameter Tuning
    9. Feature Importance
    10. Deployment
    """)

    st.subheader("🤖 Models Used")

    st.write("""
    • Logistic Regression

    • Decision Tree

    • Random Forest
    """)

    st.subheader("📈 Evaluation Metrics")

    st.write("""
    • Accuracy

    • Precision

    • Recall

    • F1 Score

    • ROC-AUC

    • Confusion Matrix
    """)