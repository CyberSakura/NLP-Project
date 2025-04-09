from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import os

# Load pre-trained BERT tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to tokenize and extract features from text
def extract_features(texts, batch_size=16):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Ensure each element is a string and remove empty strings
        batch_texts = [str(text) for text in batch_texts if str(text).strip() != '']
        
        # Tokenize input text
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Extract features using BERT
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the embeddings from the last hidden state
        embeddings = outputs.last_hidden_state
        all_embeddings.append(embeddings.cpu())  # Move to CPU to save memory
    
    # Concatenate all embeddings
    return torch.cat(all_embeddings, dim=0)

# Function to save extracted features
def save_features(features, file_path):
    torch.save(features, file_path)

# Example usage
if __name__ == "__main__":
    # Load preprocessed data
    data_dir = 'data/preprocessed'
    train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # Extract features for the entire training data
    train_texts = train_data['processed_text'].tolist()
    train_features = extract_features(train_texts)
    save_features(train_features, os.path.join(data_dir, 'train_features.pt'))
    
    # Extract features for the entire test data
    test_texts = test_data['processed_text'].tolist()
    test_features = extract_features(test_texts)
    save_features(test_features, os.path.join(data_dir, 'test_features.pt'))
    
    print("Features extracted and saved for training and test datasets.") 