import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from config import BASE_PATH, BATCH_SIZE, EPOCHS, MODEL_SAVE_PATH
from utilities import preprocess_image

#Loading text file
train_txt_path = os.path.join(BASE_PATH, "Category and Attribute Prediction Benchmark/Anno_fine/train.txt")
train_cate_path = os.path.join(BASE_PATH, "Category and Attribute Prediction Benchmark/Anno_fine/train_cate.txt")
train_attr_path = os.path.join(BASE_PATH, "Category and Attribute Prediction Benchmark/Anno_fine/train_attr.txt")

#Loading image paths and ensure they are absolute paths
with open(train_txt_path, "r", encoding="utf-8") as f:
    image_paths = [os.path.abspath(os.path.join(BASE_PATH, line.strip())) for line in f.readlines()]

#Loading category labels
with open(train_cate_path, "r", encoding="utf-8") as f:
    category_labels = np.array([int(line.strip()) for line in f.readlines()])

#Loading attribute labels
with open(train_attr_path, "r", encoding="utf-8") as f:
    attribute_labels = np.array([[int(x) for x in line.split()] for line in f.readlines()])

#limiting the dataset to x number or records, comment to take the full amount
image_paths = image_paths[:2000]
category_labels = category_labels[:2000]
attribute_labels = attribute_labels[:2000]

#Ensuring category labels are zero-indexed
category_labels = category_labels - np.min(category_labels)

#Check if unique category values
unique_categories = np.unique(category_labels)
category_mapping = {old_label: new_index for new_index, old_label in enumerate(unique_categories)}

#Applying mapping to category labels
category_labels = np.array([category_mapping[label] for label in category_labels])

num_classes = len(unique_categories)

print(f"Fixed Number of unique categories: {num_classes}")
print(f"Fixed Min category label: {np.min(category_labels)}, Max category label: {np.max(category_labels)}")
print(f"Fixed Unique category labels: {np.unique(category_labels)}")


dataset = tf.data.Dataset.from_tensor_slices((image_paths, category_labels, attribute_labels))
dataset = dataset.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

#Defining input shape (flexible for different image sizes)
input_layer = Input(shape=(None, None, 3))  # No fixed size

#Loading ResNet50 (no resizing, no final layer)
backbone = ResNet50(weights="imagenet", include_top=False, input_tensor=input_layer)

#Extracting high-level features
x = GlobalAveragePooling2D()(backbone.output)

#Category classification (Softmax)
category_output = Dense(num_classes, activation="softmax", name="category_output")(x)

#Attribute classification (Sigmoid for multi-label classification)
attribute_output = Dense(attribute_labels.shape[1], activation="sigmoid", name="attribute_output")(x)

#Define model
model = Model(inputs=input_layer, outputs=[category_output, attribute_output])

#Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss={
        "category_output": "sparse_categorical_crossentropy",  # Category classification
        "attribute_output": "binary_crossentropy"  # Multi-label classification
    },
    metrics={
        "category_output": "accuracy",
        "attribute_output": "binary_accuracy"})

model.summary()

model.fit(
    dataset,
    epochs=EPOCHS,
    verbose=1)

model.save(MODEL_SAVE_PATH)
print(f"Model saved at: {MODEL_SAVE_PATH}")