import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import nltk
import base64

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from scipy.special import softmax

# Load RoBERTa tokenizer and model
roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")


nltk.download('vader_lexicon')
sns.set_style('darkgrid')

# --- Load CSV from same folder as app.py ---
df = pd.read_csv("amazon.csv").head(500)


# --- Load Models ---
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return tokenizer, model

tokenizer, roberta_model = load_models()
vader = SentimentIntensityAnalyzer()

# --- Analysis Functions ---
def analyze_vader(text):
    return vader.polarity_scores(text)

def analyze_roberta(text):
    encoded = roberta_tokenizer(
        text,
        return_tensors='pt',
        truncation=True,        # <--- ADD THIS LINE
        max_length=512          # Optional but ensures cutoff at 512 tokens
    )
    with torch.no_grad():
        output = roberta_model(**encoded)
    scores = output.logits[0].numpy()
    scores = softmax(scores)
    labels = ['roberta_neg', 'roberta_neu', 'roberta_pos']
    return dict(zip(labels, scores))

  

def get_roberta_label(scores):
    return max(scores, key=scores.get).replace("roberta_", "")

def get_vader_label(score):
    comp = score['compound']
    return 'pos' if comp >= 0.05 else 'neg' if comp <= -0.05 else 'neu'

def sentiment_label(vader_sent, roberta_sent):
    return 'Agree' if vader_sent == roberta_sent else 'Disagree'

# --- Streamlit UI ---
st.title("AMAZON SENTIMENT ANALYSIS")

st.subheader("📄 Preview Dataset")
st.dataframe(df.head(6))


# Download button
def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="amazon_reviews.csv">📥 Download the dataset</a>'
    return href

st.markdown(file_download(df), unsafe_allow_html=True)

# Tool selection
st.subheader("🧰 Select the tool for analysis")
col1, col2, col3 = st.columns(3)

selected_tool = st.session_state.get("tool", None)

if col1.button("🔹 VADER"):
    st.session_state["tool"] = "VADER"
    selected_tool = "VADER"
if col2.button("🔸 ROBERTA"):
    st.session_state["tool"] = "ROBERTA"
    selected_tool = "ROBERTA"

if selected_tool:
    st.markdown(f"### Running {selected_tool} Analysis...")
    results = []

    for i, row in df.iterrows():
        text = str(row['reviewText'])
        overall = row['overall']
        vader_score = analyze_vader(text)
        roberta_score = analyze_roberta(text)

        vader_sent = get_vader_label(vader_score)
        roberta_sent = get_roberta_label(roberta_score)

        results.append({
            'reviewText': text,
            'overall': overall,
            'vader_pos': vader_score['pos'],
            'vader_neu': vader_score['neu'],
            'vader_neg': vader_score['neg'],
            'vader_compound': vader_score['compound'],
            **roberta_score,
            'vader_sent': vader_sent,
            'roberta_sent': roberta_sent,
            'agreement': sentiment_label(vader_sent, roberta_sent)
        })

    final_df = pd.DataFrame(results)

    # Visuals
    if selected_tool == "VADER":
        st.subheader("📊 VADER Sentiment Analysis")
        fig, ax = plt.subplots()
        sns.barplot(data=final_df, x='overall', y='vader_compound', ax=ax)
        ax.set_title("VADER Compound Score by Star Rating")
        st.pyplot(fig)

        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        sns.barplot(data=final_df, x='overall', y='vader_pos', ax=axs[0])
        axs[0].set_title('VADER Positive')
        sns.barplot(data=final_df, x='overall', y='vader_neu', ax=axs[1])
        axs[1].set_title('VADER Neutral')
        sns.barplot(data=final_df, x='overall', y='vader_neg', ax=axs[2])
        axs[2].set_title('VADER Negative')
        st.pyplot(fig)

    elif selected_tool == "ROBERTA":
        st.subheader("📊 ROBERTA Sentiment Analysis")
        fig, ax = plt.subplots()
        sns.barplot(data=final_df, x='overall', y='roberta_pos', ax=ax)
        ax.set_title("RoBERTa Positive Score by Star Rating")
        st.pyplot(fig)

        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        sns.barplot(data=final_df, x='overall', y='roberta_pos', ax=axs[0])
        axs[0].set_title('RoBERTa Positive')
        sns.barplot(data=final_df, x='overall', y='roberta_neu', ax=axs[1])
        axs[1].set_title('RoBERTa Neutral')
        sns.barplot(data=final_df, x='overall', y='roberta_neg', ax=axs[2])
        axs[2].set_title('RoBERTa Negative')
        st.pyplot(fig)

    # Store results in session state for comparison
    st.session_state["results_df"] = final_df

# Comparison Button
if st.session_state.get("results_df") is not None:
    if col3.button("Comparison"):
        st.subheader("Agreement Between VADER & RoBERTa")
        df_comp = st.session_state["results_df"]
        fig, ax = plt.subplots()
        sns.countplot(data=df_comp, x='overall', hue='agreement', ax=ax)
        ax.set_title("VADER vs RoBERTa Sentiment Agreement")
        st.pyplot(fig)

        st.subheader("🔍 Sample Comparison Results")
        st.dataframe(df_comp[['overall', 'vader_sent', 'roberta_sent', 'agreement']].head(10))


# Jupyter Notebook Button

st.markdown(
    "[📓 Click here to view the Jupyter Notebook on GitHub](https://github.com/Priya07choudhary/Amazon-sentiment-analysis/blob/main/bda.ipynb)"
)




# Description Button
if st.button("📘 Project Description"):
    st.markdown("""
    ###  PROJECT OVERVIEW: Amazon Sentiment Analysis
    
 OBJECTIVE:
To perform and compare sentiment analysis on Amazon product reviews using:

      1)VADER (a rule-based sentiment analyzer)

      2)RoBERTa (a transformer-based deep learning model)

📦 DATASET USED
Source: amazon.csv (Amazon product reviews)

Sample Size: 500 reviews (for fast testing and visualization)

Columns: reviewText, overall (star rating from 1 to 5)

🧰 LIBRARIES & TOOLS USED
Natural Language Toolkit (NLTK): Tokenization, POS tagging, Named Entity Recognition, VADER Sentiment

Transformers (Hugging Face): RoBERTa model loading and inference

Scikit-learn, Seaborn, Matplotlib: Visualization

Pandas, NumPy, tqdm: Data processing and looping

Streamlit: Web UI for live sentiment testing

STEPS PERFORMED:
1. Data Preprocessing
Loaded Amazon review dataset

Took top 500 entries

Displayed distribution of reviews across star ratings

2. Text Analysis and Tokenization
Tokenized a sample review using TreebankWordTokenizer

POS tagging using nltk.pos_tag

Named Entity Recognition using nltk.ne_chunk

3. Sentiment Analysis - VADER
Used SentimentIntensityAnalyzer to compute:

pos, neu, neg, and compound scores for each review

Merged sentiment results with original dataset

Visualized compound scores across star ratings

4. Sentiment Analysis - RoBERTa
Loaded cardiffnlp/twitter-roberta-base-sentiment

Tokenized and processed each review using RoBERTa

Applied softmax() to get probabilities for:

roberta_neg, roberta_neu, roberta_pos

Merged with VADER results

5. Visualization
Compared VADER vs RoBERTa sentiment scores per star rating

Used sns.pairplot() to visually inspect sentiment distribution

Highlighted mismatches, e.g., 5-star reviews with high negative sentiment

6. Agreement Analysis
Mapped both models' scores to sentiment labels (pos, neu, neg)

Created agreement column to show whether both models agree

Counted agreement/disagreement per star rating

Visualized with countplot

7. Most Surprising Reviews
Identified:

Most positive 1-star review

Most negative 5-star review

Displayed unexpected polarity mismatches

8. Live Sentiment Web App (Streamlit)
User enters custom text

App returns:

VADER and RoBERTa sentiment scores

Overall sentiment interpretation

Option to run this notebook directly or open it via a button

OUTCOMES:
Built a robust pipeline comparing traditional vs transformer-based sentiment analysis

Observed interesting mismatches between star ratings and inferred sentiment

Enabled real-time sentiment analysis using a friendly web interface

""")
