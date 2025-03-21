import os
import numpy as np
from tensorflow.keras.models import load_model
from utilities import preprocess_test_image
from config import BASE_PATH

# Load model from the same directory as script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "fashionfowardfinder.keras")

model = load_model(MODEL_SAVE_PATH)

TEST_IMAGE_PATH = BASE_PATH + "img/Sweet_Crochet_Blouse/img_00000070.jpg"  # Change to an actual test image path

# Preprocess the test image
test_image = preprocess_test_image(TEST_IMAGE_PATH)

# Run the model to get predictions
category_pred, attribute_pred = model.predict(test_image)

# Process category prediction
predicted_category = np.argmax(category_pred)  # Get the category with highest probability

# Process attribute predictions (Convert sigmoid outputs to binary)
attribute_pred_binary = (attribute_pred > 0.5).astype(int)  # Convert to 0 or 1

print(f"Predicted Category: {predicted_category}")
print(f"Predicted Attributes: {attribute_pred_binary}")