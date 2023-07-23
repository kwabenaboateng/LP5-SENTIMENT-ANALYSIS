import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from scipy.special import softmax


# Add description and title
st.write("""
# Sentiment Analysis App
""")


# Add image
image = st.image("images.png", width=200)


# Get user input
text = st.text_input("Type here:")
button = st.button('Analyze')

# Define the CSS style for the app
st.markdown(
"""
<style>
body {
    background-color: #f5f5f5;
}
h1 {
    color: #4e79a7;
}
</style>
""",
unsafe_allow_html=True
)


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

@st.cache_resource()
def get_model():
   # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("MrDdz/bert-base-uncased")
    return tokenizer,model
tokenizer, model = get_model()

if  button:
    text_sample = tokenizer(text, padding = 'max_length',return_tensors = 'pt')
    # print(text_sample)
    output = model(**text_sample)
    scores_ = output[0][0].detach().numpy()
    scores_ = softmax(scores_)

    labels = ['Negative','Neutral','Positive']
    scores = {l:float(s) for (l,s) in zip(labels,scores_)}
    st.write("Prediction :",scores)
