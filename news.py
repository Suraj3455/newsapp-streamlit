# NewsPulse Pro - Streamlit App
# All-in-One News App with Google Login, MongoDB, Summarization, NER, Sentiment, Clustering, and More

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from gtts import gTTS
import base64
import tempfile
import io
import spacy
from transformers import pipeline
import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import streamlit_authenticator as stauth
import re
import hashlib
import json
import uuid
import os

# ---- GLOBAL CONFIG ----
st.set_page_config(page_title="NewsPulse Pro", layout="wide", initial_sidebar_state="expanded")
st.title("📰 NewsPulse Pro")
st.markdown("A Real-Time News Analysis App with Smart Features")

# ---- GOOGLE LOGIN MOCK ----
st.sidebar.header("🔐 Login")
user_email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")
if st.sidebar.button("Login"):
    if user_email and password:
        st.session_state.user = user_email
        st.success(f"Logged in as {user_email}")
    else:
        st.error("Please enter valid credentials")

# ---- MONGO CONFIG (Optional / Commented) ----
# client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client["newspulse"]
# collection = db["articles"]

# ---- Load Models ----
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
nlp = spacy.load("en_core_web_sm")

# ---- GNEWS API ----
GNEWS_API_KEY = "2673e2462b413c6b23e6db0e295287b7"
GNEWS_ENDPOINT = "https://gnews.io/api/v4/search"

def fetch_news(keyword):
    params = {
        "q": keyword,
        "token": GNEWS_API_KEY,
        "lang": "en",
        "max": 20,
        "sortby": "publishedAt"
    }
    response = requests.get(GNEWS_ENDPOINT, params=params)
    if response.status_code == 200:
        return response.json()["articles"]
    return []

# ---- Summarize ----
def summarize_text(text):
    try:
        if len(text.split(".")) < 3:
            return text
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        return summary
    except:
        return text

# ---- Sentiment ----
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# ---- Word Cloud ----
def generate_wordcloud(texts):
    text = " ".join(texts)
    wordcloud = WordCloud(width=1000, height=600, background_color='white').generate(text)
    return wordcloud

# ---- Named Entity Recognition ----
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# ---- Topic Clustering ----
def cluster_articles(articles, k=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([a['title'] for a in articles])
    model = KMeans(n_clusters=k, random_state=42).fit(X)
    return model.labels_

# ---- Bookmark Feature ----
if 'bookmarks' not in st.session_state:
    st.session_state['bookmarks'] = []

def bookmark_article(article):
    st.session_state.bookmarks.append(article)

# ---- Main App ----
search = st.sidebar.text_input("🔍 Search News", "AI OR Election")
category = st.sidebar.selectbox("📂 Category", ["Top", "Business", "Tech", "World", "Politics", "Health"])
refresh = st.sidebar.button("🔄 Refresh News")

def render_article(article):
    st.markdown(f"### [{article['title']}]({article['url']})")
    st.image(article['image'], width=300) if article['image'] else None
    st.markdown(f"**Published**: {article['publishedAt'][:10]}")
    st.markdown(f"**Source**: {article['source']['name']}")
    summary = summarize_text(article['description'] or article['content'] or "")
    st.markdown(f"**Summary**: {summary}")
    sentiment = get_sentiment(article['title'])
    st.markdown(f"**Sentiment**: {sentiment}")
    if st.button(f"🔖 Bookmark - {article['title'][:10]}"):
        bookmark_article(article)
    entities = extract_entities(article['title'])
    st.markdown("**Entities:** " + ", ".join([e[0] for e in entities]))
    st.markdown("---")

if search and (refresh or 'last_search' not in st.session_state or search != st.session_state.last_search):
    news_data = fetch_news(search)
    if news_data:
        st.session_state.news_data = news_data
        st.session_state.last_search = search
    else:
        st.warning("No articles found")

if 'news_data' in st.session_state:
    articles = st.session_state.news_data
    st.header("🗞️ News Feed")
    for i, article in enumerate(articles):
        with st.expander(f"{i+1}. {article['title']}"):
            render_article(article)

# ---- Bookmarked Articles ----
st.sidebar.markdown("---")
st.sidebar.subheader("🔖 Bookmarked")
if st.session_state.bookmarks:
    for b in st.session_state.bookmarks:
        st.sidebar.markdown(f"[{b['title']}]({b['url']})")

# ---- Word Cloud and Sentiment Chart ----
if 'news_data' in st.session_state:
    all_titles = [a['title'] for a in articles]
    all_sentiments = [get_sentiment(a['title']) for a in articles]
    wc = generate_wordcloud(all_titles)
    st.image(wc.to_array(), caption="Word Cloud of News Titles")

    df_sent = pd.DataFrame(all_sentiments, columns=['Sentiment'])
    st.bar_chart(df_sent['Sentiment'].value_counts())

# ---- Topic Clustering ----
if 'news_data' in st.session_state:
    labels = cluster_articles(articles)
    for i in set(labels):
        st.subheader(f"🧠 Topic Cluster {i+1}")
        for j, a in enumerate(articles):
            if labels[j] == i:
                st.markdown(f"- [{a['title']}]({a['url']})")

# ---- Comments Feature ----
st.subheader("💬 Article Comments (Demo)")
if 'comments' not in st.session_state:
    st.session_state.comments = []
comment_text = st.text_input("Write a comment")
if st.button("Submit Comment"):
    st.session_state.comments.append((st.session_state.user if 'user' in st.session_state else "Anonymous", comment_text))
    st.success("Comment added!")
for u, c in st.session_state.comments:
    st.markdown(f"**{u}:** {c}")

# ---- Download Report ----
if 'news_data' in st.session_state:
    st.subheader("📥 Download Reports")
    df = pd.DataFrame(articles)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="news_report.csv">Download CSV</a>', unsafe_allow_html=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        for a in articles:
            tmp.write(f"{a['title']}\n{a['description']}\n{a['url']}\n\n".encode())
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        b64_pdf = base64.b64encode(f.read()).decode()
    st.markdown(f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="news_report.txt">Download TXT</a>', unsafe_allow_html=True)

# ---- End ----
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | NewsPulse Pro")
