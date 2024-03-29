import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import load
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import base64

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(#"stHeader"
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
setpat = r'background.jpg'
set_bg_hack(setpat)

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Load the saved model
model = load('model.joblib')
vectorizer = model[1]

# Define Streamlit app
def main():
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://unsplash.com/photos/purple-gradient-bexwsdM5BCw");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.title("Sentiment Analysis")

    # User input text area
    user_input = st.text_area("Enter your text:", "")

    # Make prediction when the user submits the input
    if st.button("Predict"):
        # Preprocess input text
        preprocessed_input = preprocess_text(user_input)
        # Vectorize the preprocessed input text
        input_vector = vectorizer.transform([preprocessed_input])
        # Make prediction using the loaded model
        prediction = model[0].predict(input_vector)[0]
        
        # Display predicted sentiment
        if prediction == 'P':
            st.header("Positive")
        else:
            st.header("Negative")

if __name__ == "__main__":
    main()
