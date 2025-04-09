# Model Implementation Guide

## Overview
This guide explains how to implement the two main components of our fake news detection system:
1. BERT-based Classification Model
2. Named Entity Recognition (NER) Integration

## Feature Description
The pre-extracted BERT features (`train_features.pt` and `test_features.pt`) are:
- Shape: (num_samples, 768) where 768 is BERT's hidden size
- Data Type: float16 (optimized for memory efficiency)
- Content: Mean-pooled BERT embeddings of article texts
- Statistics: Features are centered around 0 (mean ≈ -0.0095) with reasonable variance (std ≈ 0.288)
- Range: Values typically between -5.426 and 0.713

## 1. Classification Model Implementation

### Feature Usage Guidelines
- Load features using torch.load()
- Features are ready for direct use in classification
- No additional normalization needed
- Consider batch size based on available memory

### Model Architecture Guidelines
- Use a simple feed-forward network on top of BERT features
- Include dropout for regularization
- Output layer should have 2 neurons for binary classification
- Consider using batch normalization if needed

### Training Guidelines
- Use cross-entropy loss for binary classification
- Implement early stopping to prevent overfitting
- Monitor both training and validation metrics
- Save model checkpoints regularly

## 2. NER Integration

### NER Model Guidelines
- Use bert-large-cased-finetuned-conll03-english model
- Process articles in batches for efficiency
- Extract entities and their types (PER, ORG, LOC, etc.)
- Cache NER results to avoid reprocessing

### Feature Integration Guidelines
- Create entity-based features (counts, frequencies)
- Combine with BERT features before classification
- Ensure feature dimensions match when concatenating
- Consider feature importance for the final model

## Implementation Phases

### Phase 1: Classification Model
1. Implement basic classifier using BERT features
2. Train and validate the model
3. Document baseline performance metrics

### Phase 2: NER Integration
1. Implement NER processing pipeline
2. Create and integrate entity features
3. Retrain model with combined features
4. Compare performance with baseline

## Best Practices

1. **Data Handling**
   - Use appropriate batch sizes for your hardware
   - Implement proper validation splits
   - Monitor memory usage during processing

2. **Model Training**
   - Use learning rate scheduling
   - Implement proper logging
   - Save model and training metrics

3. **NER Processing**
   - Process articles in manageable chunks
   - Handle cases with no entities
   - Cache intermediate results

4. **Evaluation**
   - Track classification metrics (accuracy, F1, etc.)
   - Compare model performance with and without NER
   - Document improvement from NER integration

## Expected Results

1. **Classification Model**
   - Good baseline performance on fake news detection
   - Balanced performance across classes
   - Clear decision boundaries

2. **NER Integration**
   - Improved performance on entity-rich articles
   - Better handling of specific fake news patterns
   - More interpretable predictions