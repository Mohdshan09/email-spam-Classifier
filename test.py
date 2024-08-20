import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the saved model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit app
st.title("Spam Classifier")

st.write("Enter a message to classify it as spam or not spam.")

# Input text
user_input = st.text_area("Enter your message here:")

# Predict button
if st.button("Predict"):
    if user_input.strip():
        # Vectorize the input text
        input_vector = vectorizer.transform([user_input])

        # Predict using the loaded model
        prediction = model.predict(input_vector)

        # Display the result
        if prediction[0] == 1:
            st.error("This message is classified as Spam.")
        else:
            st.success("This message is classified as Not Spam.")
    else:
        st.warning("Please enter a message.")

# Additional information
st.write("This is a simple spam classifier application using a pre-trained model.")