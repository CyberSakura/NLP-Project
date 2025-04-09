# BERT Feature Documentation

## Overview
This document describes the BERT-based features extracted from the news articles dataset. The features are stored in PyTorch tensor format (.pt files) and can be used directly for training machine learning models.

## Feature Files
- `train_features.pt`: Features for training data
- `test_features.pt`: Features for test data

## Feature Details
- **Dimensions**: 
  - Training set: 35,918 samples × 768 features
  - Test set: 8,980 samples × 768 features
- **Feature Type**: BERT embeddings (mean-pooled)
- **Data Type**: torch.float32
- **Normalization**: Features are centered around 0 with reasonable variance

## How to Use the Features

### 1. Loading the Features
```python
import torch
import os

# Load the features
data_dir = 'data/preprocessed'
train_features = torch.load(os.path.join(data_dir, 'train_features.pt'))
test_features = torch.load(os.path.join(data_dir, 'test_features.pt'))
```

### 2. Loading the Labels
The features correspond to the following CSV files:
- Training labels: `data/preprocessed/train.csv`
- Test labels: `data/preprocessed/test.csv`

```python
import pandas as pd

# Load the labels
train_labels = pd.read_csv(os.path.join(data_dir, 'train.csv'))['label']
test_labels = pd.read_csv(os.path.join(data_dir, 'test.csv'))['label']
```

### 3. Example Usage with PyTorch
```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Create datasets
train_dataset = TensorDataset(train_features, torch.tensor(train_labels.values))
test_dataset = TensorDataset(test_features, torch.tensor(test_labels.values))

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Example model
class Classifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=2):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model
model = Classifier()
```

### 4. Example Usage with scikit-learn
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Convert to numpy if needed
X_train = train_features.numpy()
X_test = test_features.numpy()
y_train = train_labels.values
y_test = test_labels.values

# Train a simple classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

## Feature Generation Process
The features were generated using:
1. BERT-base-uncased model
2. Mean pooling of the last hidden state
3. Batch processing with size 8
4. Maximum sequence length of 512 tokens

## Notes
- The features are already preprocessed and ready to use
- No additional normalization is required
- The features are in CPU memory format
- Each feature vector represents the entire article's content

## Performance Tips
1. For large models, consider moving features to GPU:
```python
train_features = train_features.to('cuda')
test_features = test_features.to('cuda')
```

2. For memory efficiency, use DataLoader with appropriate batch sizes

3. Consider using feature selection or dimensionality reduction if needed:
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=100)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)
```

## Troubleshooting
If you encounter any issues:
1. Verify the file paths are correct
2. Check if the feature dimensions match your model's input requirements
3. Ensure you have enough memory for your model and batch size
4. Contact the feature generator (Kevin) for any questions 