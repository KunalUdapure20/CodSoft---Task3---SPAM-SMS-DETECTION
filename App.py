import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Ensure you have downloaded the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the model and vectorizer
with open('C:/Users/ASUS/Downloads/CodSoft/Sms Spam Classifier/model.pkl', 'rb') as model_file, \
     open('C:/Users/ASUS/Downloads/CodSoft/Sms Spam Classifier/vectorizer.pkl', 'rb') as vectorizer_file:
    model = pickle.load(model_file)
    vectorizer = pickle.load(vectorizer_file)

ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and punctuation, and stem the words
    tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation]
    
    # Join the tokens back into a single string
    return " ".join(tokens)

# Streamlit app
st.title("Email/SMS Spam Classifier")
st.write("This is a simple Streamlit application to classify messages as spam or not spam.")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = vectorizer.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
