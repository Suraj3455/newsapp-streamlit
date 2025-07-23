import streamlit as st
import requests
from textblob import TextBlob
from transformers import pipeline
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gtts import gTTS
from io import BytesIO
import nltk
import spacy
from googletrans import Translator
import os
import smtplib
from email.mime.text import MIMEText
import gc

# Setup
nltk.download("vader_lexicon")
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="NewsPulse: AI Trending & Sentiment", layout="wide")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="Falconsai/text_summarization")

@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

summarizer = load_summarizer()
vader_analyzer = load_vader()
translator = Translator()

# Email Config
EMAIL_SENDER = os.getenv("EMAIL_SENDER") or "your_email@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD") or "your_app_password"

def send_alert_email(user_email):
    if not user_email:
        return
    try:
        msg = MIMEText("⚠️ Alert: Negative sentiment spike detected in current headlines.")
        msg["Subject"] = "NewsPulse Alert"
        msg["From"] = EMAIL_SENDER
        msg["To"] = user_email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        st.success(f"📬 Alert sent to {user_email}")
    except Exception as e:
        st.error(f"❌ Email alert failed: {e}")

# NewsAPI
api_key = "380a2141d0b34d91931aa5a856a37d6f"

def fetch_news(category=None, keyword=None, max_articles=10):
    base_url = "https://newsapi.org/v2/"
    now = datetime.utcnow()
    recent_hours = 6  # Show only articles from the last 6 hours
    from_time = (now - timedelta(hours=recent_hours)).isoformat("T") + "Z"

    if keyword:
        url = (
            f"{base_url}everything?apiKey={api_key}"
            f"&q={keyword}&language=en"
            f"&from={from_time}&sortBy=publishedAt&pageSize=100"
        )
    else:
        url = (
            f"{base_url}top-headlines?apiKey={api_key}&language=en&pageSize=100"
        )
        if category:
            url += f"&category={category}"

    response = requests.get(url)
    articles = response.json().get("articles", [])

    # Manually filter if using top-headlines
    if not keyword:
        filtered_articles = []
        for article in articles:
            published_at = article.get("publishedAt")
            if published_at:
                try:
                    pub_dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                    if (now - pub_dt) <= timedelta(hours=recent_hours):
                        filtered_articles.append(article)
                except:
                    continue
        articles = filtered_articles

    return articles[:max_articles]

def analyze_sentiment_all(text):
    blob_polarity = TextBlob(text).sentiment.polarity
    vader_scores = vader_analyzer.polarity_scores(text)
    pos = round(vader_scores['pos'] * 100, 1)
    neu = round(vader_scores['neu'] * 100, 1)
    neg = round(vader_scores['neg'] * 100, 1)
    return blob_polarity, pos, neu, neg

@st.cache_data(show_spinner=False)
def generate_summary(text):
    if text and len(text) > 50:
        try:
            text = text[:600]
            summary = summarizer(text, max_length=120, min_length=40, do_sample=False)
            return summary[0]['summary_text']
        except:
            return text
    else:
        return "No summary available."

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG", "GPE", "EVENT")]

def translate_text(text, lang_code):
    try:
        return translator.translate(text, dest=lang_code).text
    except:
        return text

# Sidebar
st.sidebar.title("🔍 Filter & Search News")
category = st.sidebar.selectbox("Select News Category", ("general", "business", "sports", "technology", "entertainment"))
keyword = st.sidebar.text_input("Or enter a Search Keyword:")
lang_option = st.sidebar.selectbox("Translate Headlines To", ["English", "Hindi", "Marathi"])
lang_map = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
user_email = st.sidebar.text_input("📧 Enter your email for alerts", placeholder="you@example.com")
max_articles = st.sidebar.slider("Max articles to display", 5, 50, 10)

st.markdown("# 📰 NewsPulse: Real-Time News Trends & Sentiment AI")
st.markdown("###### Powered by NewsAPI, TextBlob, VADER, and Falcon Summarizer")

# Fetch News
articles = fetch_news(category=category, keyword=keyword, max_articles=max_articles)
sentiments_total = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
all_entities, timeline_data = [], []

for article in articles:
    text = (article.get("title") or "") + " " + (article.get("description") or "")
    if not text.strip():
        continue
    blob_polarity, pos, neu, neg = analyze_sentiment_all(text)
    timeline_data.append((article.get("publishedAt", "")[:10], blob_polarity))

    if pos > neu and pos > neg:
        sentiments_total['Positive'] += 1
    elif neg > pos and neg > neu:
        sentiments_total['Negative'] += 1
    else:
        sentiments_total['Neutral'] += 1

    entities = extract_entities(text)
    all_entities.extend(entities)

if sentiments_total['Negative'] > 5 and user_email:
    send_alert_email(user_email)

# Sentiment Display
st.markdown("## 📊 Overall Sentiment Distribution")
total_articles = sum(sentiments_total.values())
for sentiment, count in sentiments_total.items():
    percent = round((count / total_articles) * 100, 1) if total_articles else 0
    emoji = "🟢" if sentiment == "Positive" else "⚪" if sentiment == "Neutral" else "🔴"
    st.write(f"{emoji} {sentiment}: {percent}%")

# Timeline
st.markdown("## 📈 Sentiment Timeline")
if timeline_data:
    timeline_df = pd.DataFrame(timeline_data, columns=["Date", "Polarity"]).groupby("Date").mean()
    st.line_chart(timeline_df)
else:
    st.warning("No data for timeline chart.")

# Entities
if all_entities:
    st.markdown("## 🔍 Trending Entities")
    entity_counts = dict(Counter(all_entities))
    entity_df = pd.DataFrame(entity_counts.items(), columns=["Entity", "Count"]).sort_values(by="Count", ascending=False)
    st.dataframe(entity_df.head(10))

# News Results
st.markdown("## 🗞️ Latest News")
if not articles:
    st.warning("No news articles found. Try a different category or wait for API reset.")
else:
    for idx, article in enumerate(articles):
        with st.expander(f"📰 {translate_text(article.get('title', ''), lang_map[lang_option])}"):
            if article.get("urlToImage"):
                st.image(article["urlToImage"], use_container_width=True)

            text = (article.get("title") or "") + " " + (article.get("description") or "")
            blob_polarity, pos, neu, neg = analyze_sentiment_all(text)

            st.write("**Sentiment Analysis:**")
            st.write(f"🟢 Positive: {pos}% | ⚪ Neutral: {neu}% | 🔴 Negative: {neg}%")
            st.write(f"TextBlob Polarity: {round(blob_polarity*100, 1)}%")

            if st.button("📖 Show Summary", key=f"summary_{idx}"):
                summary = generate_summary(article.get("content") or article.get("description") or "")
                if lang_option != "English":
                    summary = translate_text(summary, lang_map[lang_option])
                st.success(summary)
                st.markdown("**🎧 Listen Summary:**")
                audio_fp = text_to_speech(summary, lang_map[lang_option])
                st.audio(audio_fp, format="audio/mp3")

            st.markdown(f"[🔗 Read Full Article]({article.get('url')})")
            st.caption(f"Published by: {article.get('source', {}).get('name', 'Unknown')} | Date: {article.get('publishedAt', 'N/A')}")

# WordCloud
if total_articles > 0:
    st.markdown("## ☁️ WordCloud of Headlines")
    wordcloud_text = " ".join([article.get("title", "") for article in articles])
    wordcloud = WordCloud(width=1200, height=600, background_color='white').generate(wordcloud_text)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# Download
if st.button("⬇️ Download Sentiment Report"):
    df = pd.DataFrame(articles)
    df.to_csv("sentiment_report.csv", index=False)
    with open("sentiment_report.csv", "rb") as f:
        st.download_button("Download CSV", f, file_name="report.csv")

# Bookmarks
if st.session_state.get("bookmarks"):
    st.markdown("## ⭐ Bookmarked Articles")
    for bm in st.session_state.bookmarks:
        st.markdown(f"📰 [{bm.get('title')}]({bm.get('url')}) — *{bm.get('source', {}).get('name', 'Unknown')}*")

st.markdown("---")
st.write("Made with ❤️ by Suraj Thorat | Powered by NewsAPI, HuggingFace, VADER, TextBlob")

gc.collect()
