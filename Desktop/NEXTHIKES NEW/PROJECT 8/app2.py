import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
DATA_PATH = r"C:\Users\ACER\Desktop\NEXTHIKES NEW\PROJECT 8\job_mkt.csv"
job_mkt = pd.read_csv(DATA_PATH).fillna("")

# Compute average hourly rate where applicable
job_mkt["hourly_usd"] = job_mkt[["hourly_low", "hourly_high"]].mean(axis=1)
job_mkt.loc[job_mkt["is_hourly"] == False, "hourly_usd"] = None  # only for hourly jobs

# TF-IDF on job titles
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(job_mkt["title"])

# Streamlit UI
st.set_page_config(page_title="Job Recommender", layout="wide")
st.title("🤖 Job Recommendation Engine")

# Sidebar filters
st.sidebar.header("Filters")
all_countries = ["All"] + sorted(job_mkt["country"].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Select Country", all_countries)

# User input
query = st.text_input("Enter your skills or job role:", "Python Developer")
top_n = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Get Recommendations"):
    query_vec = vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sim_scores.argsort()[::-1][:top_n * 3]  # get more, filter later

    results = job_mkt.iloc[top_idx][
        ["job_id", "title", "country", "hourly_low", "hourly_high", "hourly_usd", "link"]
    ]

    # Apply country filter
    if selected_country != "All":
        results = results[results["country"] == selected_country]

    # Limit to top_n after filtering
    results = results.head(top_n)

    # Rename for display
    results = results.rename(columns={
        "job_id": "Job ID",
        "title": "Job Title",
        "country": "Country",
        "hourly_low": "Min Rate (USD/hr)",
        "hourly_high": "Max Rate (USD/hr)",
        "hourly_usd": "Avg Rate (USD/hr)",
        "link": "Job Link"
    })

    st.subheader("🔎 Recommended Jobs")
    st.dataframe(results, use_container_width=True)
