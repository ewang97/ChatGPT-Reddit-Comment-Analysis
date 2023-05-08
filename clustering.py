
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

#Uses the elbow method to find number of clusters that best represents the data (low error distance from centers) without overfitting k
def optimal_k(k_range, tf_idf):
    param = range(1,k_range)
    res_k = 1
    sse = []
    for k in param:
        model = KMeans(n_clusters= k, init="k-means++", random_state=10, max_iter=100, n_init=1)
        model.fit(tf_idf)
        sse.append(model.inertia_)
        print(f'Fitting for {k} clusters')
    
    max_diff = 0
    for i in range(0,len(sse)-1):
        decrease_sse = sse[i] - sse[i+1]
        if decrease_sse > max_diff:
            res_k = i + 1

    return res_k

#Use optimal cluster number k to label the data
def cluster_comments(k_max, tf_idf):
    print('Calculating optimal k...')
    k = optimal_k(k_max,tf_idf)

    print('Generating cluster labels...')
    model = KMeans(n_clusters=k, init="k-means++", random_state=20, max_iter=2000, n_init=1)
    y_predicted = model.fit_predict(tf_idf)

    return y_predicted

if __name__ == "__main__":
    cwd = os.getcwd()
    cleaned_df = pd.read_csv(cwd + '/processed/cleaned_comments.csv')

    #Generate TF_IDF representation of comments as a parameter for clustering
    vectorizer = TfidfVectorizer(
                                lowercase=True,
                                ngram_range = (1,3),
                                stop_words = "english"                              
                            )

    tf_idf = vectorizer.fit_transform(cleaned_df.comment_body)
    clustered_df = cleaned_df.copy()
    
    #Generate clustered copy of dataset
    clustered_df['cluster'] = cluster_comments(5,tf_idf)
    
    print('Exporting labeled dataset...')
    #Export to data folder
    path = cwd + "/processed/clustered_comments.csv"
    clustered_df.to_csv(path)

