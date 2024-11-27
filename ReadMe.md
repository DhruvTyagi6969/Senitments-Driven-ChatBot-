Code Overview

This Python script implements a complete machine learning pipeline for sentiment analysis using TensorFlow and Streamlit. Here's what it does:
Data Preprocessing:
    a. Cleans and preprocesses raw text data.
    b. Performs tokenization, removal of stopwords, and lemmatization.
    c. Uses TF-IDF vectorization for numerical representation of text.
Model Development:
    a. Builds a neural network using TensorFlow's Sequential API.
    b. Trains the model with preprocessed text data and binary sentiments (positive/negative).
Deployment:
    a. Integrates the trained model with a Streamlit app to predict sentiment from user input.
Model Persistence:
    a. Saves the trained model and the TF-IDF vectorizer using joblib for reuse.
Interactive Interface:
    a. Provides an interactive web interface to input reviews and display predictions
