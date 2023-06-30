import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # returns the text by converting all the letters to lowercase
    text = nltk.word_tokenize(text)  # returns the list of words in SMS

    # removing the special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):


#pre-processing
    transform_sms = transform_text(input_sms)

#vectorize
    vector_input = tfidf.transform([transform_sms])

#prediction
    result = model.predict(vector_input)[0]

#display
    if result==1:
        st.header('Spam')
    else:
        st.header("Not Spam")