import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Path to your CSV ---
DATA_PATH = r"C:\Users\ACER\Desktop\NEXTHIKES NEW\PROJECT 8\job_mkt.csv"

# --- Load Data ---
def load():
    job_mkt = pd.read_csv(DATA_PATH)

    # Convert dates
    job_mkt['published_date'] = pd.to_datetime(job_mkt['published_date'], errors='coerce', utc=True)
    job_mkt['month'] = pd.PeriodIndex(job_mkt['published_date'], freq='M')

    # Detect remote jobs
    job_mkt['remote'] = job_mkt['title'].fillna("").str.contains(
        r"\b(?:remote|wfh|work[-\s]?from[-\s]?home|telecommute|hybrid)\b",
        case=False, regex=True
    )

    return job_mkt

job_mkt = load()
# --- Streamlit Page Config ---
st.set_page_config(page_title="Job Market Dashboard", layout="wide")
st.title("📊 Job Market Dynamics Dashboard")

# --- Sidebar Filters ---
st.sidebar.header("Filters")
selected_country = st.sidebar.selectbox(
    "Select Country", ["All"] + sorted(job_mkt['country'].dropna().unique().tolist())
)
selected_year = st.sidebar.selectbox(
    "Select Year", ["All"] + sorted(job_mkt['published_date'].dt.year.dropna().unique().tolist())
)

# Apply filters
data = job_mkt.copy()
if selected_country != "All":
    data = data[data['country'] == selected_country]
if selected_year != "All":
    data = data[data['published_date'].dt.year == int(selected_year)]

# --- Trends over time ---
st.subheader("📈 Job Postings Over Time")
monthly_counts = data.groupby('month').size()

fig, ax = plt.subplots(figsize=(10, 5))
monthly_counts.plot(ax=ax, marker='o')
ax.set_ylabel("Number of Jobs")
ax.set_xlabel("Month")
st.pyplot(fig)

# --- Salary trends ---
st.subheader("💰 Average Hourly USD Rates Over Time")
if "hourly_usd" in data.columns:
    monthly_salary = data.groupby('month')['hourly_usd'].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_salary.plot(ax=ax, marker='s', color='green')
    ax.set_ylabel("Avg Hourly Rate (USD)")
    ax.set_xlabel("Month")
    st.pyplot(fig)
else:
    st.warning("⚠️ No `hourly_usd` column found in the dataset.")

# --- Country distribution ---
st.subheader("🌍 Top Countries by Job Postings")
if "country" in data.columns:
    top_countries = data['country'].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_countries.index, y=top_countries.values, ax=ax)
    ax.set_ylabel("Job Count")
    ax.set_xlabel("Country")
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.warning("⚠️ No `country` column found in the dataset.")



