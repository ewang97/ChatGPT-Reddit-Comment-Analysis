# ChatGPT-Reddit-Comment-Analysis

-----------------------

K-Means Clustering and Sentiment Analysis on Reddit comments related to ChatGPT in order to gauge current public perception on the latest AI Language Model. Data used includes comments that were extracted from 4 popular Reddit sub-communities whose focus has been around ChatGPT in early 2023. Data was downloaded via https://www.kaggle.com/datasets/armitaraz/chatgpt-reddit

Visualizations were then created via Jupyter Notebook to better present findings based on the clustering used.


Installation
----------------------

### Download the data/Reproduce

* Clone this repo to your computer.
* Get into the folder using `cd chatgpt-text-analysis`.
* For use of updated data, simply redownload and name the data 'chatgpt-reddit-comments.csv', then place in 'raw_data' folder

### Install the requirements
 
* Install the requirements using `pip install -r requirements.txt`.
    * Python 3.

Usage
-----------------------

* Run 'python preprocess.py' to clean the data
* Run 'python clustering.py' to produce clusters/labels based on similarites. TF-IDF matrix representation of comments was used for K-Means algorithm
* Run all modules in 'visualization.ipynb' to get visuals for the clusters as well as sentiment analysis.
    * Sentiment analysis was done using the compound score metric specified by VADER (https://pypi.org/project/vaderSentiment/).

Extended/To Do
-------------------------

* Other vector representations can be experimented with to see if clustering yields different results (Bag of words, word2vec)
* Further work on sentiment can be added - see if there is any correlation between positive/negative and how the clusters are formed
