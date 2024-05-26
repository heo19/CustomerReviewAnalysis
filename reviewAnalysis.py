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
import random   
from nltk.corpus import stopwords  # remove meaningless words.
from nltk.tokenize import word_tokenize  # For tokenizing text into words. Splitting text into words.
import re  # regex

# Download necessary NLTK data files for stopwords and tokenization
nltk.download('stopwords') # words that are meaningless
nltk.download('punkt') # tokenizaer data.

# List of user agents to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
]

#Crawling Amazon reviews using BeautifulSoup
def get_amazon_reviews(url):
    # Set headers to simulate a browser visit
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
    }

    reviews = [] # list to store reviews
    page = 1 # starting page number

    while page <= 20:
        # Send a GET request to the Amazon URL
        response = requests.get(url + f'/ref=cm_cr_getr_d_paging_btm_next_{page}?pageNumber={page}', headers=headers)
        if response.status_code != 200: # status 200 means the request was not susccessful.
            print("request failed")
            break

        # Get HTML using BeautifulSoup.
        soup = BeautifulSoup(response.content, 'html.parser')
        # Get review blocks using data-hook attribute.
        review_blocks = soup.find_all('div', {'data-hook': 'review'})
        print(f"Found {len(review_blocks)} review blocks on page {page}")
        #If no Review, break the loop
        if not review_blocks:
            print("No reviews")
            break
        
        #Repeat for each review block to get the data.
        for block in review_blocks:
            review_text = block.find('span', {'data-hook': 'review-body'}).get_text().strip()
            r_rating_element = block.select_one("i.review-rating")
            r_rating = r_rating_element.text.replace("out of 5 stars", "") if r_rating_element else None
            reviews.append((review_text, int(float(r_rating))))  # Store review text and rating
    

        # Move to the next page
        page += 1

    return reviews

amazon_url = "https://www.amazon.com/Kafka-on-Shore-Haruki-Murakami-audiobook/dp/B00EAP9IPO/ref=sr_1_1?crid=2SJ4BF9ZWDXOC&dib=eyJ2IjoiMSJ9.qwjK8ZqjsMNT30fFhM6Dc53BogejxRiwSiljKlnyu20gp1BOEUnHIUNF2B5NPs3pbL3LYoI-XZmmgy3LbO94sj4tKLfhrg3UlzI_-MdoMiDYeRrHO_tsGIQzKsJg9rAxNumOolTsZFpRuCDz1535fTgRRq0zyYZ9L2zIhjNSrQN8yiDWMuP17GvBCaibTLICy4RCj01KFHxmC6cbRFJTS3iZHmdVHDBBXuNH7aVx8-4.BiiRep5dkL3k1tp2gKCPN_np0EMeUEc-nuVv0MmGSi8&dib_tag=se&keywords=Kafka+on+the+Shore&qid=1716647258&sprefix=kafka+on+the+shore%2Caps%2C244&sr=8-1" # Url of the product page
reviews = get_amazon_reviews(amazon_url) # Get reviews from the Amazon product page
df = pd.DataFrame(reviews, columns=['review', 'rating']) # Create a DataFrame from the reviews

# Check if reviews are fetched correctly
if not reviews:
    print("No reviews found.")
else:
    print(f"Fetched {len(reviews)} reviews.")


#Get Crawled Data and clean it
def clean_text(text):
    text = re.sub(r'\W', ' ', text) # remove special characters
    text = re.sub(r'\s+', ' ', text) # remove extra spaces
    text = text.lower() # convert text to lowercase
    text = word_tokenize(text) # tokenize text into words
    text = [word for word in text if word not in stopwords.words('english')] # remove stopwords
    return ' '.join(text)  #combine words into a string

df['cleaned_text'] = df['review'].apply(clean_text) # Clean the reviews using the clean_text function

# Check if DataFrame is not empty after cleaning
if df['cleaned_text'].isnull().all():
    print("All cleaned texts are empty.")
else:
    print("Cleaned texts are processed.")

#load the pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text data using the BERT tokenizer
def tokenize_data(text):
    return tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")

df['input_ids'] = df['cleaned_text'].apply(lambda x: tokenize_data(x)['input_ids'][0]) # apply the tokenize_data function to the cleaned_text column
df['attention_mask'] = df['cleaned_text'].apply(lambda x: tokenize_data(x)['attention_mask'][0]) # apply the tokenize_data function to the cleaned_text column

input_ids = torch.stack(df['input_ids'].tolist()) # stack the input_ids into a tensor
attention_masks = torch.stack(df['attention_mask'].tolist()) # stack the attention_masks into a tensor
labels = torch.tensor(df['rating'].values) # convert the ratings into a tensor

# Create a ReviewFeature class to store input_ids, attention_mask, and label
# This debug the error of having  TypeError: vars() argument must have dict attribute
# ensure that each feature in your dataset is an instance of a class, rather than a primitive data type or a tuple.
class ReviewFeature:
    def __init__(self, input_ids, attention_mask, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label = label

#dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels) # Create a PyTorch dataset from the input data

#Create Dataset from the ReviewFeature class
dataset = [ReviewFeature(input_id, attention_mask, label) for input_id, attention_mask, label in zip(input_ids, attention_masks, labels)]

#Fine-tuning the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

#Split the dataset into training and testing sets
train_size = 0.8
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(train_size * len(dataset)), len(dataset) - int(train_size * len(dataset))])
#print(train_dataset[0]) # Debugging

# Set training arguments for the Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Create a Trainer instance to train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Make predictions on the test dataset
predictions, _, _ = trainer.predict(test_dataset)
predictions = np.argmax(predictions, axis=1)

# Get the true labels from the test dataset
test_labels = [label.label for label in test_dataset]

# Evaluate the model performance
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions, average='weighted')
recall = recall_score(test_labels, predictions, average='weighted')
f1 = f1_score(test_labels, predictions, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')