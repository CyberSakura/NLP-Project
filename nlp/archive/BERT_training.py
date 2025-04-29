# ================================
# Hugging Face Login
# ================================
from huggingface_hub import login

# Use own token
login(token="YOUR_HUGGING_FACE_TOKEN")

# ================================
# Import Libraries
# ================================
import torch
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
from datasets import Dataset
import evaluate

# ================================
# Load Data
# ================================
data_dir = 'data/preprocessed'
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# Convert to lists for processing and ensure all text is string
train_texts = [str(text) for text in train_df['processed_text'].tolist()]
test_texts = [str(text) for text in test_df['processed_text'].tolist()]

# Convert labels to proper integers
train_labels = [int(label) for label in train_df['label'].tolist()]
test_labels = [int(label) for label in test_df['label'].tolist()]

# Create datasets
train_dataset = Dataset.from_dict({
    'text': train_texts,
    'label': train_labels
})
test_dataset = Dataset.from_dict({
    'text': test_texts,
    'label': test_labels
})

# ================================
# Load BERT Model and Tokenizer
# ================================
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ================================
# Prepare Dataset
# ================================
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Tokenize the datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# ================================
# Training Setup
# ================================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# ================================
# Load NER Model (keeping existing NER implementation)
# ================================
ner_pipe = pipeline(
    "ner",
    model="Jean-Baptiste/roberta-large-ner-english",
    tokenizer="Jean-Baptiste/roberta-large-ner-english",
    aggregation_strategy="simple"
)

def extract_entity_features(texts):
    features = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            features.append([0, 0, 0, 0])
            continue
        ents = ner_pipe(text[:512])
        counts = {"PER": 0, "ORG": 0, "LOC": 0, "MISC": 0}
        for ent in ents:
            label = ent["entity_group"]
            if label in counts:
                counts[label] += 1
        features.append([counts["PER"], counts["ORG"], counts["LOC"], counts["MISC"]])
    return np.array(features, dtype=np.float32)

# ================================
# Train BERT Model
# ================================
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# ================================
# Evaluate the Model
# ================================
predictions = trainer.predict(tokenized_test)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

print("\nEvaluation: Fine-tuned BERT")
print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
print(f"Precision: {precision_score(labels, preds):.4f}")
print(f"Recall: {recall_score(labels, preds):.4f}")
print(f"F1 Score: {f1_score(labels, preds):.4f}")
print("\nClassification Report:\n", classification_report(labels, preds, target_names=["REAL", "FAKE"]))