# ================================
# Import Libraries
# ================================
import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ================================
# Load and Prepare Data
# ================================
train_df = pd.read_csv("data/preprocessed/train.csv")
test_df = pd.read_csv("data/preprocessed/test.csv")

# Handle NaN values in processed_text
train_df['processed_text'] = train_df['processed_text'].fillna('')  # Replace NaN with empty string
test_df['processed_text'] = test_df['processed_text'].fillna('')    # Replace NaN with empty string

# First, let's check the data types and content
print("Training data info:")
print(train_df.info())
print("\nSample of labels:", train_df['label'].head())
print("\nUnique labels:", train_df['label'].unique())

# Then modify the label conversion accordingly
# If labels are already numeric (0 and 1), we don't need to convert
if train_df['label'].dtype == 'object':  # if labels are strings
    train_df['label'] = train_df['label'].str.lower().map({'fake': 1, 'true': 0})
    test_df['label'] = test_df['label'].str.lower().map({'fake': 1, 'true': 0})
else:  # if labels are already numeric
    train_df['label'] = train_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)

# ================================
# Prepare BERT Input
# ================================
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        truncation=True, 
        padding='max_length', 
        max_length=512
    )

# Convert to HuggingFace datasets
train_dataset = Dataset.from_dict({
    'text': train_df['processed_text'].tolist(),
    'label': train_df['label'].astype(int).tolist()
})

test_dataset = Dataset.from_dict({
    'text': test_df['processed_text'].tolist(),
    'label': test_df['label'].astype(int).tolist()
})

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# ================================
# Initialize BERT Model
# ================================
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# ================================
# Training Arguments
# ================================
training_args = TrainingArguments(
    output_dir="models/bert_finetuned",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100
)

# ================================
# Metrics Function
# ================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions),
        'f1': f1_score(labels, predictions)
    }

# ================================
# Initialize Trainer
# ================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

# ================================
# Train and Evaluate
# ================================
print("Starting BERT fine-tuning...")
trainer.train()

# Evaluate
print("\nEvaluating fine-tuned BERT model...")
eval_results = trainer.evaluate()

print("\nFinal Evaluation Results:")
for metric, value in eval_results.items():
    if metric.startswith('eval_'):
        print(f"{metric[5:]}: {value:.4f}")

# Save the model
model.save_pretrained("models/bert_finetuned_final")
tokenizer.save_pretrained("models/bert_finetuned_final")