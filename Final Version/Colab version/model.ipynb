"""
Train a multi-task model for clothing category classification and attribute prediction
using ResNet50 as the backbone.

This script loads image paths and labels, prepares a TensorFlow dataset, builds a
dual-head CNN model using ResNet50, and trains it. The model simultaneously predicts
a single clothing category (out of 50) and multiple attributes.

Dependencies:
    - TensorFlow
    - NumPy
    - pickle
    - config.py (contains constants like BASE_PATH, EPOCHS, etc.)
    - utilities.py (contains preprocess_image function)
"""

import os
import numpy as np
import pickle
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
import tensorflow as tf
from config import *
from utilities import preprocess_image

# Load file paths for training images and labels
train_txt_path = os.path.join(BASE_PATH, "Category and Attribute Prediction Benchmark/Anno_fine/train.txt")
train_cate_path = os.path.join(BASE_PATH, "Category and Attribute Prediction Benchmark/Anno_fine/train_cate.txt")
train_attr_path = os.path.join(BASE_PATH, "Category and Attribute Prediction Benchmark/Anno_fine/train_attr.txt")

# Read image paths and convert to absolute paths
with open(train_txt_path, "r", encoding="utf-8") as f:
    image_paths = [os.path.abspath(os.path.join(BASE_PATH, line.strip())) for line in f.readlines()]

# Read category labels (1-based indexing from file)
with open(train_cate_path, "r", encoding="utf-8") as f:
    category_labels = np.array([int(line.strip()) for line in f.readlines()])

# Read attribute labels (multi-label format)
with open(train_attr_path, "r", encoding="utf-8") as f:
    attribute_labels = np.array([[int(x) for x in line.split()] for line in f.readlines()])

# Optional: Subset data for quicker training/debugging
# image_paths = image_paths[:2000]
# category_labels = category_labels[:2000]
# attribute_labels = attribute_labels[:2000]

# Convert category labels to zero-based indices
category_labels = category_labels - 1

# Prepare the TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((image_paths, category_labels, attribute_labels))
dataset = dataset.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Define the model input with flexible image size
input_layer = Input(shape=(None, None, 3))

# Load pretrained ResNet50 without the top classification layer
backbone = ResNet50(weights="imagenet", include_top=False, input_tensor=input_layer)

# Global average pooling to reduce feature maps to a vector
x = GlobalAveragePooling2D()(backbone.output)

# Output head for category classification (50 classes, softmax)
category_output = Dense(50, activation="softmax", name="category_output")(x)

# Output head for attribute classification (multi-label, sigmoid)
attribute_output = Dense(attribute_labels.shape[1], activation="sigmoid", name="attribute_output")(x)

# Build the final multi-task model
model = Model(inputs=input_layer, outputs=[category_output, attribute_output])

# Compile the model with appropriate losses and metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss={
        "category_output": "sparse_categorical_crossentropy",
        "attribute_output": "binary_crossentropy"
    },
    metrics={
        "category_output": "accuracy",
        "attribute_output": "binary_accuracy"
    })

# Print model architecture
model.summary()

# Train the model
history = model.fit(
    dataset,
    epochs=EPOCHS,
    verbose=1
)

# Save the trained model
model.save(MODEL_SAVE_PATH)
print(f"Model saved at: {MODEL_SAVE_PATH}")

# Save training history
with open(MODEL_HISTORY_SAVE_PATH, "wb") as f:
    pickle.dump(history.history, f)

print(f"Training history saved at: {MODEL_HISTORY_SAVE_PATH}")
