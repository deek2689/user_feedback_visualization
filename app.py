import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from gensim.models import LdaModel
from wordcloud import WordCloud
import numpy as np
import textstat
from math import pi
import torch

# Load Data
def load_data():
    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"], key="file_uploader_unique_1")  # Unique key here
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)
    return None


# Ensure that nltk stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function to clean text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text






# Visualization Functions
def visualize_feedback_distribution(data):
    st.subheader("Feedback Distribution by Rank")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='rank', data=data, palette='coolwarm')
    plt.title("Distribution of Feedback")
    st.pyplot(plt)

def visualize_top_articles(data):
    st.subheader("Top Articles with Positive Feedback and Weak Acceptance")

    # Filter articles with rank 1 (positive feedback) and rank 0 (weak acceptance)
    positive_feedback_articles = data[data['rank'] == 1][['title', 'rank']]
    weak_acceptance_articles = data[data['rank'] == 0][['title', 'rank']]

    # Combine both positive feedback and weak acceptance articles
    combined_articles = pd.concat([positive_feedback_articles, weak_acceptance_articles])

    # Get the top 5 articles from both categories
    top_articles = combined_articles['title'].value_counts().head(5)

    # Plotting the data
    plt.figure(figsize=(6, 4))
    top_articles.plot(kind='barh', color=['#4CAF50' if rank == 1 else '#FF6347' for rank in combined_articles['rank'][:len(top_articles)]])

    plt.xlabel("Count")
    plt.title("Top Articles with Positive Feedback and Weak Acceptance")

    # Display the plot in Streamlit
    st.pyplot(plt)

def get_top_articles_with_positive_feedback(data):
    # Filter articles with positive feedback (rank = 1) and weak acceptance (rank = 0)
    feedback_articles = data[data['rank'].isin([0, 1])][['title', 'rank', 'link']]  # Assuming 'link' is a column with URLs

    # Return the filtered data
    return feedback_articles

# Step 2: Visualize the categories of the top articles
 #Step 2: Visualize the categories of the top articles
def visualize_article_categories(data):
    st.subheader("Major Categories Explored from Top Articles with Positive Feedback and Weak Acceptance")

    # Categories (These should match the labels you are using for classification)
    categories = ['Technology', 'Finance', 'Health & Wellness', 'Food', 'Lifestyle', 
                  'Entertainment', 'Sports', 'Science', 'Business', 'Politics']

    # Get top articles based on positive feedback and weak acceptance
    top_articles = get_top_articles_with_positive_feedback(data)  # Assuming this function returns both positive and weak feedback

    if top_articles.empty:
        st.write("No articles with positive feedback or weak acceptance found.")
        return

    # Load pre-trained BERT model for sequence classification (zero-shot classification approach)
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Step 1: Classify each top article into predefined categories
    article_category_scores = {category: {'positive': 0, 'weak': 0} for category in categories}

    for _, article in top_articles.iterrows():
        article_content = extract_content_from_url(article['link'])  # Extract content from article URL
        
        if article_content:
            result = classifier(article_content, candidate_labels=categories)
            # Get the top predicted category for each article
            top_category = result['labels'][0]
            if article['rank'] == 1:
                article_category_scores[top_category]['positive'] += 1  # Positive feedback
            else:
                article_category_scores[top_category]['weak'] += 1  # Weak acceptance

    # Step 2: Extract separate scores
    positive_scores = [article_category_scores[category]['positive'] for category in categories]
    weak_scores = [article_category_scores[category]['weak'] for category in categories]

    # Normalize both scores to the same scale
    max_value = max(max(positive_scores), max(weak_scores))
    positive_scores = [x / max_value for x in positive_scores]
    weak_scores = [x / max_value for x in weak_scores]

    # Number of categories
    num_categories = len(categories)

    # Radar plot setup
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()

    # Close the plot by connecting last point to first point
    positive_scores += positive_scores[:1]
    weak_scores += weak_scores[:1]
    angles += angles[:1]

    # Create the radar plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Plot positive feedback (with higher peaks)
    ax.plot(angles, positive_scores, linewidth=2, linestyle='solid', label='Positive Feedback', color='green')
    ax.fill(angles, positive_scores, alpha=0.4, color='green')

    # Plot weak acceptance (with smaller spikes)
    ax.plot(angles, weak_scores, linewidth=2, linestyle='dashed', label='Weak Acceptance', color='red')
    ax.fill(angles, weak_scores, alpha=0.2, color='red')

    # Labels and title
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title("Category Distribution of Top Articles with Positive Feedback vs Weak Acceptance", size=14, color='blue', y=1.1)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Display the plot in Streamlit
    st.pyplot(fig)

# Helper function to extract article content from URL (using BeautifulSoup)
# Improved content extractor with multiple fallback methods
# Extract Content from URL
def extract_content_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        print(content)

        if not content:
            return ""

        return content
    except Exception:
        return ""



# Function to calculate readability score
def calculate_readability_score(text):
    if not text or len(text.split()) < 30:
        return None
    return textstat.flesch_reading_ease(text)

# Visualization for Readability vs User Acceptance
def visualize_readability_vs_acceptance(data):
    st.subheader("Readability Score vs User Acceptance")

    readability_scores = []
    acceptance_scores = []

    for _, row in data.iterrows():
        article_content = extract_content_from_url(row['link'])
        if article_content:
            readability_score = calculate_readability_score(article_content)
            if readability_score is not None:
                readability_scores.append(readability_score)
                acceptance_scores.append(row['rank'])

    plot_data = pd.DataFrame({
        'Readability Score': readability_scores,
        'Acceptance Score': acceptance_scores
    })

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x='Readability Score',
        y='Acceptance Score',
        data=plot_data,
        hue='Acceptance Score',
        palette={1: 'green', 0: 'orange', -1: 'red'},
        s=100
    )

    plt.title("Readability Score vs User Acceptance")
    plt.xlabel("Readability Score (Higher = Easier to Read)")
    plt.ylabel("User Acceptance")
    plt.axvline(x=60, color='blue', linestyle='--', alpha=0.6, label='Standard Readability (60)')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.legend(title="User Acceptance")

    st.pyplot(plt)

    # Add download button
    csv_data = plot_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Readability Data as CSV",
        data=csv_data,
        file_name="readability_vs_acceptance.csv",
        mime='text/csv'
    )



# Streamlit App
def main():
    st.title("User Feedback Visualization Dashboard")
    data = load_data()

    if data is not None:
        st.write("### Preview of Data")
        st.dataframe(data.head())

        visualize_feedback_distribution(data)
        visualize_top_articles(data)
        #visualize_major_topics(data)
        visualize_readability_vs_acceptance(data)
        #visualize_readability_vs_acceptance2(data)
        visualize_article_categories(data)
        #visualize_readability_vs_acceptance(data)
        #visualize_recommendation_score_over_time(data)
        #perform_sentiment_analysis(data)
    else:
        st.warning("Please upload a valid file to continue.")

    # Add Footer with Image and Name
    #st.markdown("---")
    #col1, col2 = st.columns([0.8, 0.2])
    #with col2:
        #st.image("/home/enas/Downloads/1694441388613 (1) (1).jpg", width=100)
        #st.markdown("**Deeksha**")

if __name__ == "__main__":
    main()

