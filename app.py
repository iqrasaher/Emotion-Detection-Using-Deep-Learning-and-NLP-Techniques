# imports====================
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.stem import PorterStemmer
import re

#load models=================
model = pickle.load(open('Logistic_Regression.pkl','rb'))
lb = pickle.load(open('LabelEncoder.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))
#custom function==============
stopwords = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]"," ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)
def prediction(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized= tfidf_vectorizer.transform([cleaned_text])

    predicted_label = model.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label = np.max(model.predict(input_vectorized))
    return predicted_emotion,label
#app==========================
st.title("Six NLP Emotions Detection App")
st.write(['Joy','Fear','Love','Anger','Sadness','Surprise'])
input_text =st.text_input("Write your text here")

if st.button("predict"):
    predicted_emotion,label=prediction(input_text)
    st.write("Predicted Emotion:",predicted_emotion)
    st.write("Predicted Label:",label)



