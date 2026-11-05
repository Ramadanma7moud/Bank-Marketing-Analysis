import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Data Insights",
    page_icon="📊",
    layout="wide"
)

# Page Title
st.title("📊 Bank Customer Data Insights")

st.write(
    "Explore customer data and discover patterns "
    "that influence term deposit subscriptions."
)

st.divider()

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("bank.csv")

df = load_data()

# Dataset Overview
st.subheader("📁 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", f"{len(df):,}")

with col2:
    st.metric("Total Features", len(df.columns) - 1)

with col3:
    st.metric("Target Variable", "deposit")

# Display Dataset
with st.expander("View Dataset"):
    st.dataframe(df.head(10), use_container_width=True)
    # Subscription Distribution

st.divider()
st.subheader("📊 Subscription Distribution")

subscription_counts = df['deposit'].value_counts()

col1, col2 = st.columns(2)

with col1:
    st.metric("Subscribed Customers", subscription_counts['yes'])

with col2:
    st.metric("Not Subscribed Customers", subscription_counts['no'])

st.write("### Customer Subscription Chart")

st.bar_chart(subscription_counts)
# Subscription Rate by Job

st.divider()
st.subheader("💼 Subscription Rate by Job")

job_subscription = pd.crosstab(
    df['job'],
    df['deposit'],
    normalize='index'
) * 100

st.write("### Subscription Rate (%)")

st.bar_chart(job_subscription['yes'].sort_values(ascending=False))

st.write("### Detailed Job Statistics")

st.dataframe(
    job_subscription.round(2),
    use_container_width=True
)
# Subscription Rate by Age Group

st.divider()
st.subheader("👥 Subscription Rate by Age Group")

df['age_group'] = pd.cut(
    df['age'],
    bins=[17, 25, 35, 45, 55, 65, 100],
    labels=[
        '18-25', '26-35', '36-45',
        '46-55', '56-65', '65+'
    ]
)

age_subscription = pd.crosstab(
    df['age_group'],
    df['deposit'],
    normalize='index'
) * 100

st.write("### Subscription Rate (%)")

st.bar_chart(age_subscription['yes'])

st.write("### Age Group Statistics")

st.dataframe(
    age_subscription.round(2),
    use_container_width=True
)
# Subscription Rate by Education

st.divider()
st.subheader("🎓 Subscription Rate by Education")

education_subscription = pd.crosstab(
    df['education'],
    df['deposit'],
    normalize='index'
) * 100

st.write("### Subscription Rate (%)")

st.bar_chart(
    education_subscription['yes'].sort_values(ascending=False)
)

st.write("### Education Statistics")

st.dataframe(
    education_subscription.round(2),
    use_container_width=True
)
# Subscription Rate by Marital Status

st.divider()
st.subheader("💍 Subscription Rate by Marital Status")

marital_subscription = pd.crosstab(
    df['marital'],
    df['deposit'],
    normalize='index'
) * 100

st.write("### Subscription Rate (%)")

st.bar_chart(
    marital_subscription['yes'].sort_values(ascending=False)
)

st.write("### Marital Status Statistics")

st.dataframe(
    marital_subscription.round(2),
    use_container_width=True
)
# Subscription Rate by Account Balance

st.divider()
st.subheader("💰 Subscription Rate by Account Balance")

df['balance_group'] = pd.cut(
    df['balance'],
    bins=[-float('inf'), 0, 1000, 5000, 10000, float('inf')],
    labels=[
        'Negative',
        '0-1000',
        '1001-5000',
        '5001-10000',
        'Above 10000'
    ]
)

balance_subscription = pd.crosstab(
    df['balance_group'],
    df['deposit'],
    normalize='index'
) * 100

st.bar_chart(balance_subscription['yes'])

st.dataframe(
    balance_subscription.round(2),
    use_container_width=True
)