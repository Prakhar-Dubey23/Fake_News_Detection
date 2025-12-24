import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Minor Project using Python & Machine Learning")

@st.cache_data
def load_data():
    df = pd.read_csv(
        "news.csv",
        encoding="utf-8",
        engine="python",
        on_bad_lines="skip"
    )
    df = df[["text", "label"]]
    df.dropna(inplace=True)
    return df

df = load_data()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

vectorizer = CountVectorizer(stop_words="english", max_features=8080)
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

news_input = st.text_area("Enter News Text")

if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        cleaned = clean_text(news_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == "FAKE":
            st.error("🟥 FAKE NEWS")
        else:
            st.success("🟩 REAL NEWS")
