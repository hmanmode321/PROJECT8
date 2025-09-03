from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
DATA_PATH = r"C:\Users\ACER\Desktop\NEXTHIKES NEW\PROJECT 8\job_mkt.csv"
job_mkt = pd.read_csv(DATA_PATH).fillna("")

# Compute average hourly rate
job_mkt["hourly_usd"] = job_mkt[["hourly_low", "hourly_high"]].mean(axis=1)
job_mkt.loc[job_mkt["is_hourly"] == False, "hourly_usd"] = None

# TF-IDF on job titles
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(job_mkt["title"])

# Create FastAPI instance
app = FastAPI(title="Job Recommendation API")

@app.post("/recommend_jobs/")
def recommend_jobs(query: str, country: str = None, top_n: int = 5):
    query_vec = vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sim_scores.argsort()[::-1]

    results = job_mkt.iloc[top_idx][
        ["job_id", "title", "country", "hourly_low", "hourly_high", "hourly_usd", "link"]
    ]

    # Filter by country if provided
    if country:
        results = results[results["country"].str.lower() == country.lower()]

    return {
        "query": query,
        "country_filter": country,
        "recommendations": results.head(top_n).to_dict(orient="records"),
    }
