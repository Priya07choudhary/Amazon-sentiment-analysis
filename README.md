#AMAZON SENTIMENT ANALYSIS

This project aims to classify the sentiment of Amazon customer reviews using two different approaches:
Click on this link to view this project: https://amazon-sentiment-analysis-priyachoudhary.streamlit.app/

VADER (from the Natural Language Toolkit)
RoBERTa Transformer (via Hugging Face's transformers library)
We also compare the results from both sentiment analysis methods to highlight their differences and insights.

Dataset
We used a dataset containing approximately 5000 Amazon customer reviews across multiple product categories such as books, electronics, and kitchen appliances. Each review includes a star rating from 1 to 5.

Preprocessing
Before applying sentiment analysis, we performed the following preprocessing steps:

Removed non-alphabetic characters
Converted all text to lowercase
Removed stop words
Applied stemming to reduce words to their root form
Sentiment Analysis
We applied sentiment analysis using:

VADER (Valence Aware Dictionary and sEntiment Reasoner)

A rule-based model that uses a predefined lexicon of words and their sentiment scores.
RoBERTa Transformers

A deep learning-based, state-of-the-art language model that understands the context and nuances of natural language.
We compared the sentiment scores produced by VADER and RoBERTa and found notable differences. These variations were further analyzed to understand the strengths and limitations of each approach.

Tools & Libraries
Anaconda
Jupyter Notebook
Python libraries:
pandas, numpy
seaborn, matplotlib
nltk, re
transformers
Conclusion
This project demonstrates how to perform sentiment analysis on Amazon product reviews using both traditional rule-based and advanced deep learning methods. The differences in sentiment scores emphasize the importance of understanding the assumptions, limitations, and applications of each sentiment analysis tool.

This serves as a strong foundation for anyone exploring sentiment analysis, NLP, or customer review analytics.
