import streamlit as st
import joblib

# Load the trained model
model = joblib.load('bank_deposit_model.pkl')

# Page configuration
st.set_page_config(
    page_title="Bank Deposit Prediction",
    page_icon="🏦",
    layout="centered"
)

# Main title
st.title("🏦 Bank Term Deposit Prediction")

st.write(
    "Predict whether a customer is likely to subscribe "
    "to a bank term deposit."
)

st.divider()

st.subheader("Customer Information")
# Customer Input Fields

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Personal Information")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    job = st.selectbox(
        "Job",
        [
            "admin.", "blue-collar", "entrepreneur",
            "housemaid", "management", "retired",
            "self-employed", "services", "student",
            "technician", "unemployed", "unknown"
        ]
    )

    marital = st.selectbox(
        "Marital Status",
        ["married", "single", "divorced"]
    )

    education = st.selectbox(
        "Education",
        ["primary", "secondary", "tertiary", "unknown"]
    )

    balance = st.number_input(
        "Account Balance",
        value=0
    )

    default = st.selectbox(
        "Credit in Default",
        ["no", "yes"]
    )

with col2:
    st.markdown("### 🏦 Banking & Campaign Information")

    housing = st.selectbox(
        "Housing Loan",
        ["no", "yes"]
    )

    loan = st.selectbox(
        "Personal Loan",
        ["no", "yes"]
    )

    contact = st.selectbox(
        "Contact Type",
        ["cellular", "telephone", "unknown"]
    )

    day = st.number_input(
        "Last Contact Day",
        min_value=1,
        max_value=31,
        value=15
    )

    month = st.selectbox(
        "Last Contact Month",
        [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ]
    )

    campaign = st.number_input(
        "Number of Contacts During This Campaign",
        min_value=1,
        value=1
    )

    pdays = st.number_input(
        "Days Since Previous Contact",
        min_value=-1,
        value=-1
    )

    previous = st.number_input(
        "Number of Previous Contacts",
        min_value=0,
        value=0
    )

    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        ["unknown", "failure", "other", "success"]
    )
import pandas as pd

# Prepare customer data
input_data = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'default': [default],
    'balance': [balance],
    'housing': [housing],
    'loan': [loan],
    'contact': [contact],
    'day': [day],
    'month': [month],
    'campaign': [campaign],
    'pdays': [pdays],
    'previous': [previous],
    'poutcome': [poutcome]
})
st.divider()

if st.button("Predict Subscription", use_container_width=True):

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    yes_index = list(model.classes_).index('yes')
    yes_probability = probability[0][yes_index]

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction[0] == 'yes':
        st.markdown(f"""
        <div class="result-card success-card">
            <h2>✅ Likely to Subscribe</h2>
            <p>The customer is predicted to subscribe to a term deposit.</p>
            <h2>{yes_probability:.2%}</h2>
            <p>Subscription Probability</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-card warning-card">
            <h2>⚠️ Unlikely to Subscribe</h2>
            <p>The customer is predicted not to subscribe to a term deposit.</p>
            <h2>{yes_probability:.2%}</h2>
            <p>Subscription Probability</p>
        </div>
        """, unsafe_allow_html=True)
    st.progress(float(yes_probability))
   
    # Custom CSS

st.markdown("""
<style>
    .stApp {
        background-color: #f4f7fb;
    }

    h1 {
        color: #17365D;
        text-align: center;
        font-weight: 800;
    }

    h2, h3 {
        color: #17365D;
    }

    div.stButton > button {
        background-color: #17365D;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px 25px;
        font-weight: bold;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #24558a;
        color: white;
        .result-card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin: 15px 0;
    background-color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border: 1px solid #dce3ed;
}

.result-card h2 {
    margin-bottom: 10px;
}

.result-card p {
    font-size: 18px;
    color: #555;
}

.success-card {
    border-left: 6px solid #16a34a;
}

.warning-card {
    border-left: 6px solid #d97706;
}
    }
</style>
""", unsafe_allow_html=True)
