#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import regex as re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import nltk
import tensorflow as tf

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load and clean data
df = pd.read_csv("lol.csv")
df['clean_text'] = df['review'].apply(lambda x: re.sub("<.*?>", "", x))
df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'[^\w\s]', "", x))
df['clean_text'] = df['clean_text'].str.lower()

# Tokenize and remove stopwords
stop_words = set(stopwords.words('english'))
df['tokenized_text'] = df['clean_text'].apply(lambda x: word_tokenize(x))
df['filter_text'] = df['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])

# Lemmatize
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
df['lemma_text'] = df['filter_text'].apply(lambda x: [lemma.lemmatize(word) for word in x])

# Prepare data for TF-IDF
X = df['lemma_text'].apply(lambda x: " ".join(x))  # Convert list of words back to sentences for TF-IDF
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Build and compile model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train_tfidf.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")  # Use sigmoid for binary classification
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train_tfidf, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test_tfidf, y_test_encoded))

# Save model and TF-IDF vectorizer
model.save('model.h5')
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(le, 'label_encoder.pkl')  # Save label encoder

# Streamlit app for sentiment prediction
import streamlit as st

# Load the model and TF-IDF vectorizer
model = tf.keras.models.load_model('model.h5')
tfidf = joblib.load('tfidf.pkl')
le = joblib.load('label_encoder.pkl')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def predict_sentiment(review):
    # Clean and preprocess the review
    cleaned_review = re.sub('<.*?>', '', review)  # Remove HTML tags
    cleaned_review = re.sub(r'[^\w\s]', '', cleaned_review)  # Remove punctuation
    cleaned_review = cleaned_review.lower()  # Convert to lowercase
    tokenized_review = word_tokenize(cleaned_review)  # Tokenize text
    filtered_review = [word for word in tokenized_review if word not in stop_words]  # Remove stopwords
    stemmed_review = [stemmer.stem(word) for word in filtered_review]  # Stem words
    
    # Transform the review using TF-IDF vectorizer
    tfidf_review = tfidf.transform([' '.join(stemmed_review)])
    
    # Get the prediction from the model
    sentiment_prediction = model.predict(tfidf_review)
    
    # Adjust prediction logic based on model output
    return "Positive" if sentiment_prediction[0][0] > 0.5 else "Negative"

# Streamlit app layout
st.title('Sentiment Analysis')
review_to_predict = st.text_area('Enter your review here:')
if st.button('Predict Sentiment'):
    if review_to_predict.strip():  # Ensure there is input text
        predicted_sentiment = predict_sentiment(review_to_predict)
        st.write("Predicted Sentiment:", predicted_sentiment)
    else:
        st.write("Please enter a review to classify.")


# In[ ]:





# In[ ]:




