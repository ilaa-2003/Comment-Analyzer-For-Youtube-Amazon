import streamlit as st
from googleapiclient.discovery import build
from textblob import TextBlob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------- CONFIG ----------- #
YOUTUBE_API_KEY = 'AIzaSyCpl-ZEBJC1uZkRJ07MlxrIQaNYZ-t8dZQ'  # 🔁 Replace with your actual key
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# ----------- FUNCTIONS ----------- #
def get_youtube_comments(video_id, max_comments=100):
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=1000,
            textFormat="plainText",
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments[:max_comments]

def analyze_sentiment(texts):
    results = []
    for text in texts:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        sentiment = (
            "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
        )
        results.append({
            "comment": text,
            "polarity": polarity,
            "sentiment": sentiment
        })
    return pd.DataFrame(results)

def plot_sentiment_distribution(df, title="Sentiment Distribution"):
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='sentiment', palette='viridis', ax=ax)
    ax.set_title(title)
    return fig

# ----------- STREAMLIT APP ----------- #
st.title("🧠 Sentiment Analyzer for YouTube & Amazon Reviews")

# ----------- YouTube Comments ----------- #
st.header("🎥 YouTube Comment Sentiment Analysis")
video_id = st.text_input("Enter YouTube Video ID")

if video_id:
    with st.spinner("Fetching YouTube comments..."):
        comments = get_youtube_comments(video_id, max_comments=1000)
        if comments:
            df_youtube = analyze_sentiment(comments)
            st.success(f"Fetched and analyzed {len(df_youtube)} comments.")
            st.dataframe(df_youtube)
            st.pyplot(plot_sentiment_distribution(df_youtube, "YouTube Comments"))
            st.markdown(f"**📊 Average Sentiment**: `{df_youtube['sentiment'].mode()[0]}`")
        else:
            st.warning("No comments found or invalid video ID.")

# ----------- Amazon Reviews ----------- #
st.header("🛒 Amazon Review Sentiment Analysis")
uploaded_file = st.file_uploader("Upload Amazon Review CSV", type="csv")

if uploaded_file:
    try:
        df_amazon_raw = pd.read_csv(uploaded_file)
        if 'reviews.text' not in df_amazon_raw.columns:
            st.error("The uploaded file must contain a 'reviews.text' column.")
        else:
            reviews = df_amazon_raw['reviews.text'].dropna().tolist()
            df_amazon = analyze_sentiment(reviews)
            st.success(f"Analyzed {len(df_amazon)} reviews.")
            st.dataframe(df_amazon)
            st.pyplot(plot_sentiment_distribution(df_amazon, "Amazon Reviews"))
            st.markdown(f"**📊 Average Sentiment**: `{df_amazon['sentiment'].mode()[0]}`")
    except Exception as e:
        st.error(f"Error processing file: {e}")