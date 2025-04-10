"""
Model Evaluation Report: Classification Confidence and Precision Visualization

This script evaluates the performance of a dual-head neural network for clothing classification:
- Predicts categories (single-label) and attributes (multi-label) on a validation set
- Computes top-k accuracy for categories, hamming loss, and detailed classification metrics for attributes
- Visualizes:
    - Top-3 category accuracy as a bar chart
    - Per-attribute precision as a bar plot

Dependencies:
    - TensorFlow
    - NumPy
    - Scikit-learn
    - Matplotlib
    - Seaborn
    - Custom config and utility functions
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, hamming_loss, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from config import *
from utilities import *

# Load validation file paths
train_txt_path = os.path.join(BASE_PATH, "Category and Attribute Prediction Benchmark/Anno_fine/val.txt")
train_cate_path = os.path.join(BASE_PATH, "Category and Attribute Prediction Benchmark/Anno_fine/val_cate.txt")
train_attr_path = os.path.join(BASE_PATH, "Category and Attribute Prediction Benchmark/Anno_fine/val_attr.txt")

# Load image paths (absolute)
with open(train_txt_path, "r", encoding="utf-8") as f:
    image_paths = [os.path.abspath(os.path.join(BASE_PATH, line.strip())) for line in f.readlines()]

# Load expected category labels (1-based indexing)
with open(train_cate_path, "r", encoding="utf-8") as f:
    expected_categories = np.array([int(line.strip()) for line in f.readlines()])

# Load expected attribute labels (multi-hot format)
with open(train_attr_path, "r", encoding="utf-8") as f:
    expected_attributes = np.array([[int(x) for x in line.split()] for line in f.readlines()])

# Load trained model
model = load_model(MODEL_SAVE_PATH)

# Predict categories and attributes
predicted_categories = []
predicted_attributes = []
raw_category_preds = []  # Keep raw softmax outputs for top-k evaluation

for img_path in image_paths:
    img = preprocess_test_image(img_path)
    category_pred, attr_pred = model.predict(img, verbose=0)

    raw_category_preds.append(category_pred[0])
    predicted_categories.append(convert_to_category_output(category_pred))
    predicted_attributes.append(convert_to_attribute_output(attr_pred).flatten())

# Convert predictions to numpy arrays
predicted_categories = np.array(predicted_categories)
predicted_attributes = np.array(predicted_attributes)
raw_category_preds = np.array(raw_category_preds)

def top_k_accuracy(y_true, y_pred_probs, k=3):
    """
    Computes top-k categorical accuracy.
    
    Args:
        y_true (array): True labels (1-based).
        y_pred_probs (array): Prediction probabilities.
        k (int): Number of top predictions to consider.
        
    Returns:
        float: Top-k accuracy score.
    """
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:] + 1
    return np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])

# ===== Reporting =====

print("=== Category Classification Report ===")
print(f"Top-3 Accuracy: {top_k_accuracy(expected_categories, raw_category_preds, 3):.4f}")

print("\n=== Attribute Classification Report ===")
print("Hamming Loss:", hamming_loss(expected_attributes, predicted_attributes))
print(classification_report(expected_attributes, predicted_attributes, digits=4))

# ===== Visualization =====

top3_accuracy = top_k_accuracy(expected_categories, raw_category_preds, k=3)
attribute_precisions = precision_score(expected_attributes, predicted_attributes, average=None)

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [1, 4]})

# Left: Top-3 Accuracy
sns.barplot(x=["Top-3 Category Accuracy"], y=[top3_accuracy], palette=["#2a9d8f"], ax=axes[0])
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Top-3 Category Accuracy")

# Right: Per-Attribute Precision
sns.barplot(x=list(range(len(attribute_precisions))), y=attribute_precisions, palette="viridis", ax=axes[1])
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("Attribute Index")
axes[1].set_ylabel("Precision")
axes[1].set_title("Per-Attribute Precision")
axes[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()
