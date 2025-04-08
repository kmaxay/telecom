import pandas as pd
import joblib
import streamlit as st

# Load saved model
try:
    model = joblib.load("xgb_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found. Train the model first!")
    st.stop()

# Custom CSS for styling
st.markdown(
    """
    <style>
        .main { background-color: #f0f2f6; }
        .title { text-align: center; color: #4A90E2; }
        .subheader { text-align: center; }
        .stButton button { background-color: #4CAF50; color: white; font-size: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Title
st.markdown("<h1 class='title'>📡 Telecommunication Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subheader'>Enter Customer Information Below</h3>", unsafe_allow_html=True)

# Define input fields matching model features
wanted_columns = [
    "account.length", "voice.messages", "intl.mins", "intl.calls", "eve.mins", "day.mins", 
    "night.calls", "night.mins", "customer.calls", "intl.plan_yes"
]

# Layout using columns
with st.form("churn_form"):
    col1, col2 = st.columns(2)
    with col1:
        account_length = st.number_input("📅 Account Length (days)", min_value=0, format="%.0f", value=0)
        voice_messages = st.number_input("📩 Number of Voice Messages", min_value=0, format="%.0f", value=0)
        intl_mins = st.number_input("🌍 International Minutes", min_value=0.0, format="%.2f")
        intl_calls = st.number_input("📞 Number of International Calls", min_value=0, format="%.0f", value=0)
        eve_mins = st.number_input("🌆 Evening Minutes", min_value=0.0, format="%.2f")
    
    with col2:
        day_mins = st.number_input("🌞 Day Minutes", min_value=0.0, format="%.2f")
        night_calls = st.number_input("🌙 Number of Night Calls", min_value=0, format="%.0f", value=0)
        night_mins = st.number_input("🌙 Night Minutes", min_value=0.0, format="%.2f")
        customer_calls = st.number_input("👨‍💼 Customer Service Calls", min_value=0, format="%.0f")
        intl_plan = st.radio("🌍 International Plan?", options=["Yes", "No"], horizontal=True)
    
    submit_button = st.form_submit_button("🚀 Predict Churn")

# Process input and make prediction
if submit_button:
    intl_plan_yes = 1 if intl_plan == "Yes" else 0  # Convert to binary

    # Prepare input data
    input_data = pd.DataFrame([{
        "account.length": account_length,
        "voice.messages": voice_messages,
        "intl.mins": intl_mins,
        "intl.calls": intl_calls,
        "eve.mins": eve_mins,
        "day.mins": day_mins,
        "night.calls": night_calls,
        "night.mins": night_mins,
        "customer.calls": customer_calls,
        "intl.plan_yes": intl_plan_yes,
    }])

    # Ensure order matches training columns
    input_data = input_data[wanted_columns]

    # Predict using the trained model
    prediction = model.predict(input_data)

    # Display result with better UI
    churn_result = "🛑 YES, Customer is likely to Churn!" if prediction[0] == 1 else "✅ NO, Customer is unlikely to Churn."
    st.markdown(f"<h2 style='text-align: center; color: {'red' if prediction[0] == 1 else 'green'}'>{churn_result}</h2>", unsafe_allow_html=True)
