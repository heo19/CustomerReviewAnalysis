#Importing necessary libraries
import pandas as pd  # Data manipulation and analysis library
import numpy as np  # Numerical computing library
import requests  # For making HTTP requests to fetch web content (crawling)
from bs4 import BeautifulSoup  # For parsing HTML and extracting data from web pages (crawling)
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # For evaluating the performance of the model
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments  # For using pre-trained BERT model and training it
import torch
import nltk
from nltk.corpus import stopwords  # remove meaningless words.
from nltk.tokenize import word_tokenize  # For tokenizing text into words. Splitting text into words.
import re  # regex
from flask import Flask, request, jsonify  # For creating a web server and API

# Download necessary NLTK data files for stopwords and tokenization
nltk.download('stopwords') # words that are meaningless
nltk.download('punkt') # tokenizaer data.


#Crawling Amazon reviews using BeautifulSoup
def get_amazon_reviews(url):
    # Set headers to simulate a browser visit
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    reviews = [] # list to store reviews
    page = 1 # starting page number

    while True:
        # Send a GET request to the Amazon URL
        response = requests.get(url + f'/ref=cm_cr_getr_d_paging_btm_next_{page}?pageNumber={page}', headers=headers)
        if response.status_code != 200: # status 200 means the request was not susccessful.
            break

        # Get HTML using BeautifulSoup.
        soup = BeautifulSoup(response.content, 'html.parser')
        # Get review blocks using data-hook attribute.
        review_blocks = soup.find_all('div', {'data-hook': 'review'})

        #If no Review, break the loop
        if not review_blocks:
            break
        
        #Repeat for each review block to get the data.
        for block in review_blocks:
            review_text = block.find('span', {'data-hook': 'review-body'}).get_text().strip()
            reviews.append(review_text)

        # Move to the next page
        page += 1

    return reviews

amazon_url = "YOUR_AMAZON_PRODUCT_URL" # Url of the product page
reviews = get_amazon_reviews(amazon_url) # Get reviews from the Amazon product page
df = pd.DataFrame(reviews, columns=['review']) # Create a DataFrame from the reviews
