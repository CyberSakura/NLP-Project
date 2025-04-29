import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def load_data(fake_path, true_path):
    """
    Load and combine fake and true news datasets
    """
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    # Add label column (0 for true, 1 for fake)
    fake_df['label'] = 1
    true_df['label'] = 0
    
    # Combine datasets
    df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
    return df

def preprocess_text(text):
    """
    Preprocess text using spaCy and NLTK
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization using spaCy
    doc = nlp(text)
    
    # Lemmatization and stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    return processed_text

def prepare_dataset(df, test_size=0.2, random_state=42):
    """
    Prepare dataset by preprocessing text and splitting into train/test sets
    """
    # Preprocess title and text
    print("Preprocessing titles...")
    df['processed_title'] = df['title'].apply(preprocess_text)
    print("Preprocessing main text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split into train and test sets
    X = df[['processed_title', 'processed_text']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test 