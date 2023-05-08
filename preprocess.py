import os

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
nltk.download ('wordnet')
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


import re

stop_words = set(stopwords.words('english'))
#stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def clean_comments(comment):
    #Normalize text
    comment = comment.lower()

    #Remove special characters/URLS and '@' tags and non numbers/letters
    comment = re.sub(r"(@\[A-Za-z0-9]+)|([^A-Za-z])|(\w+:\/\/\S+)|^rt|http.+?", " ", comment)

    #Remove linebreaks
    comment = re.sub(r"\n", " ",comment)
    
    #Remove non-words/single characters
    comment = re.sub(r"(^| ).(( ).)*( |$)", " ", comment)

    #Remove stopwords and lemmatize
    word_tokens = word_tokenize(comment)
    clean_token_list = []
    for word_token in word_tokens:
        if word_token not in stop_words:
            clean_token_list.append(lemmatizer.lemmatize(word_token,pos = 'v'))
    
    clean_comment = " ".join(clean_token_list)
    
    return clean_comment

if __name__ == "__main__":
    cwd = os.getcwd()
    df = pd.read_csv(cwd + '/raw_data/chatgpt-reddit-comments.csv')
    
    #Check for/Remove NULL values from data set
    df.dropna(inplace=True)

    #Generate cleaned copy of dataset
    cleaned_df = df.copy()
    cleaned_df['comment_body'] = df.comment_body.apply(lambda x: clean_comments(x))\
    
    #Finally remove any rows that have become empty due to cleaning
    cleaned_df['comment_body'].replace('', np.nan, inplace=True)
    cleaned_df.dropna(inplace=True)
    
    #Export to data folder
    path = cwd + "/processed/cleaned_comments.csv"
    cleaned_df.to_csv(path)
