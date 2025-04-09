import torch
import os

# Define the directory where features are saved
data_dir = 'nlp/data/preprocessed'

# Load the extracted features
train_features = torch.load(os.path.join(data_dir, 'train_features.pt'))
test_features = torch.load(os.path.join(data_dir, 'test_features.pt'))

# Check the shape of the features
print("Train Features Shape:", train_features.shape)
print("Test Features Shape:", test_features.shape)

# Optionally, inspect a sample of the features
print("Sample Train Feature:", train_features[0]) 