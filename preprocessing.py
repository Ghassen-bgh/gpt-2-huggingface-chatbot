# preprocessing.py
from nltk.tokenize import word_tokenize

def preprocess_input(user_input):
    tokens = word_tokenize(user_input.lower())
    return tokens
