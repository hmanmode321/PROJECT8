# Job Market Analysis and Recommendation System
# Technologies: Python, Pandas, Scikit-Learn, TensorFlow, Flask, Docker

# ========== Step 1: Correlation Between Job Title Keywords and Salaries ==========

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

# Load dataset
jobs_df = pd.read_csv("jobs.csv")  # columns: ['job_title', 'salary', 'location', ...]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=100)
X_keywords = vectorizer.fit_transform(jobs_df['job_title'])

# Regression to analyze correlation
model = LinearRegression()
model.fit(X_keywords, jobs_df['salary'])

# Correlation analysis
correlation = pd.Series(model.coef_, index=vectorizer.get_feature_names_out())
correlation.sort_values(ascending=False).head(10).plot(kind='bar', title='Top Keywords Correlated with High Salary')
plt.tight_layout()
plt.show()

# ========== Step 2: Identify Emerging Job Categories ==========

jobs_df['date_posted'] = pd.to_datetime(jobs_df['date_posted'])
jobs_df['month'] = jobs_df['date_posted'].dt.to_period('M')
category_trend = jobs_df.groupby(['month', 'job_category']).size().unstack(fill_value=0)
category_trend.tail(12).plot(figsize=(12, 6), title='Emerging Job Categories')
plt.tight_layout()
plt.show()

# ========== Step 3: Predict High-Demand Job Roles ==========
from sklearn.ensemble import RandomForestRegressor

job_counts = jobs_df.groupby(['month', 'job_title']).size().reset_index(name='count')
job_pivot = job_counts.pivot(index='month', columns='job_title', values='count').fillna(0)

X = job_pivot.shift(1).dropna()
y = job_pivot.loc[X.index]
model = RandomForestRegressor()
model.fit(X, y)
prediction = model.predict([X.iloc[-1]])
pd.Series(prediction[0], index=X.columns).sort_values(ascending=False).head(10).plot(kind='bar', title='Forecasted High-Demand Roles')
plt.tight_layout()
plt.show()

# ========== Step 4: Compare Hourly Rates Across Countries ==========

country_salary = jobs_df.groupby('country')['hourly_rate'].mean()
country_salary.plot(kind='barh', figsize=(10, 8), title='Average Hourly Rate by Country')
plt.tight_layout()
plt.show()

# ========== Step 5: Job Recommendation Engine ==========
from sklearn.neighbors import NearestNeighbors

vectorizer = TfidfVectorizer(max_features=100)
X_job = vectorizer.fit_transform(jobs_df['job_title'])
nbrs = NearestNeighbors(n_neighbors=5).fit(X_job)

# Recommendation function
def recommend_jobs(input_title):
    input_vec = vectorizer.transform([input_title])
    distances, indices = nbrs.kneighbors(input_vec)
    return jobs_df.iloc[indices[0]][['job_title', 'location', 'salary']]

print(recommend_jobs("data scientist"))

# ========== Step 6: Job Market Dynamics Dashboard ==========
# Use Streamlit (save as streamlit_app.py)

# streamlit_app.py
import streamlit as st
st.title("Monthly Job Trends")
st.line_chart(category_trend)

# ========== Step 7: Remote Work Trends ==========

remote_jobs = jobs_df[jobs_df['remote'] == True]
remote_trend = remote_jobs.groupby(jobs_df['date_posted'].dt.to_period('M')).size()
remote_trend.plot(kind='line', title='Remote Work Trend Over Time')
plt.tight_layout()
plt.show()

# ========== Step 8: Predict Future Job Market Trends ==========
from sklearn.linear_model import Ridge

monthly_postings = jobs_df.groupby('month').size()
X_time = pd.Series(range(len(monthly_postings))).values.reshape(-1, 1)
y_time = monthly_postings.values

future_model = Ridge()
future_model.fit(X_time, y_time)
future_x = [[i] for i in range(len(monthly_postings), len(monthly_postings)+6)]
predicted = future_model.predict(future_x)

plt.plot(monthly_postings.index.astype(str), y_time, label='Actual')
plt.plot(pd.period_range(monthly_postings.index[-1]+1, periods=6, freq='M').astype(str), predicted, label='Forecast')
plt.legend()
plt.title("Future Job Market Forecast")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========== Docker Setup Files (save separately) ==========
# Dockerfile (for Flask or Streamlit app)
'''
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "streamlit_app.py"]
'''

# requirements.txt
'''
pandas
scikit-learn
streamlit
matplotlib
seaborn
plotly
'''

# docker-compose.yml
'''
version: '3'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - .:/app
'''

# README.md (instructions)
'''
1. Build Docker Image: docker build -t job-analyzer .
2. Run App: docker-compose up
3. Visit: http://localhost:8501
'''
