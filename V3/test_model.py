import numpy as np
from tensorflow.keras.models import load_model
from utilities import *
from config import *
import pickle
import matplotlib.pyplot as plt

model = load_model(MODEL_SAVE_PATH)

# Preprocess the test image
test_image = preprocess_test_image(TEST_IMAGE_PATH)

# Run the model to get predictions
category_pred, attribute_pred = model.predict(test_image)

print(f"Predicted Attributes before : {attribute_pred}")
print(f"Predicted Category before : {category_pred}")
# Process category prediction
predicted_category =  convert_to_category_output(category_pred)

# Process attribute predictions (Convert sigmoid outputs to binary)
predicted_attributes = convert_to_attribute_output(attribute_pred) 

print(f"Predicted Category: {predicted_category}")
print(f"Predicted Attributes: {predicted_attributes}")

category_name = CATEGORY_NAMES[predicted_category]
attribute_names = [ATTRIBUTE_NAMES[i + 1] for i, val in enumerate(predicted_attributes[0]) if val == 1]

keyword = get_search_keyword_from_prediction(predicted_category, predicted_attributes)

print(f"Predicted Category: {category_name['desc']} | Predicted Attributes: {', '.join([a['desc'] for a in attribute_names])}")

print(f"search keyword: {keyword}")


with open(MODEL_HISTORY_SAVE_PATH, "rb") as f:
    history_data = pickle.load(f)

epochs = range(1, len(history_data['loss']) + 1)

plt.figure(figsize=(14, 6))

# --- Losses ---
plt.subplot(1, 2, 1)
plt.plot(epochs, history_data['loss'], label='Total Loss', color='red', linewidth=2)
plt.plot(epochs, history_data['category_output_loss'], label='Category Loss', color='orange', linestyle='--')
plt.plot(epochs, history_data['attribute_output_loss'], label='Attribute Loss', color='purple', linestyle='--')
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# --- Accuracies ---
plt.subplot(1, 2, 2)
plt.plot(epochs, history_data['category_output_accuracy'], label='Category Accuracy', color='blue', linewidth=2)
plt.plot(epochs, history_data['attribute_output_binary_accuracy'], label='Attribute Accuracy', color='green', linestyle='--')
plt.title('Training Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
