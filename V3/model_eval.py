import os
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from utilities import preprocess_test_image
from sklearn.metrics import accuracy_score, classification_report, hamming_loss, precision_score
import numpy as np
from config import *
from utilities import *


# Load text file paths
train_txt_path = os.path.join(BASE_PATH, "Category and Attribute Prediction Benchmark/Anno_fine/val.txt")
train_cate_path = os.path.join(BASE_PATH, "Category and Attribute Prediction Benchmark/Anno_fine/val_cate.txt")
train_attr_path = os.path.join(BASE_PATH, "Category and Attribute Prediction Benchmark/Anno_fine/val_attr.txt")

# Load image paths and ensure they are absolute paths
with open(train_txt_path, "r", encoding="utf-8") as f:
    image_paths = [os.path.abspath(os.path.join(BASE_PATH, line.strip())) for line in f.readlines()]

# Load category labels
with open(train_cate_path, "r", encoding="utf-8") as f:
    expected_categories = np.array([int(line.strip()) for line in f.readlines()])

# Load attribute labels
with open(train_attr_path, "r", encoding="utf-8") as f:
    expected_attributes = np.array([[int(x) for x in line.split()] for line in f.readlines()])
    
model = load_model(MODEL_SAVE_PATH)

# Collect predictions
predicted_categories = []
predicted_attributes = []

raw_category_preds = []  

for img_path in image_paths:
    img = preprocess_test_image(img_path)

    category_pred, attr_pred = model.predict(img, verbose=0)

    raw_category_preds.append(category_pred[0])  # Save softmax as 1D array

    category_pred = convert_to_category_output(category_pred)
    attr_pred = convert_to_attribute_output(attr_pred)

    predicted_categories.append(category_pred)
    predicted_attributes.append(attr_pred.flatten())


predicted_categories = np.array(predicted_categories)
predicted_attributes = np.array(predicted_attributes)
raw_category_preds = np.array(raw_category_preds)

def top_k_accuracy(y_true, y_pred_probs, k=3):
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:] + 1  # shift to 1-based
    return np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])


print("=== Category Classification Report ===")
print(f"Top-3 Accuracy: {top_k_accuracy(expected_categories, raw_category_preds, 3):.4f}")


print("\n=== Attribute Classification Report ===")
print("Hamming Loss:", hamming_loss(expected_attributes, predicted_attributes))
print(classification_report(expected_attributes, predicted_attributes, digits=4))


# Compute values
top3_accuracy = top_k_accuracy(expected_categories, raw_category_preds, k=3)
attribute_precisions = precision_score(expected_attributes, predicted_attributes, average=None)

# Create side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [1, 4]})

# Plot Top-3 Accuracy
sns.barplot(x=["Top-3 Category Accuracy"], y=[top3_accuracy], palette=["#2a9d8f"], ax=axes[0])
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Top-3 Category Accuracy")

# Plot Per-Attribute Precision
sns.barplot(x=list(range(len(attribute_precisions))), y=attribute_precisions, palette="viridis", ax=axes[1])
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("Attribute Index")
axes[1].set_ylabel("Precision")
axes[1].set_title("Per-Attribute Precision")
axes[1].tick_params(axis='x', rotation=90)

# Adjust layout
plt.tight_layout()
plt.show()