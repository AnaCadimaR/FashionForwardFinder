import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from config import *

def preprocess_image(img_path, category_label, attribute_label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Decode image

    # Convert to float32
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Get image dimensions
    shape = tf.shape(img)
    height, width = shape[0], shape[1]

    # Pad images to the largest size
    img = tf.image.resize_with_pad(img, MAX_HEIGHT, MAX_WIDTH)

    return img, (tf.cast(category_label, tf.int32), tf.cast(attribute_label, tf.float32))

def preprocess_test_image(img_path):
    """Loads and preprocesses an image for model inference."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3) 
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_pad(img, MAX_HEIGHT, MAX_WIDTH) 
    img = tf.expand_dims(img, axis=0)  
    return img

def get_search_keyword_from_image() -> str:
    """Extracts search keyword from the image filename."""
    
    model = load_model(MODEL_SAVE_PATH)
    test_image = preprocess_test_image(TEST_IMAGE_PATH)
    category_pred, attribute_pred = model.predict(test_image)
    
    predicted_index =  np.argmax(category_pred, axis=1)[0]  
    predicted_category = predicted_index + 1
    
    predicted_attributes = (attribute_pred > ATTRIBUTES_SENSIBILITY).astype(int)  # Convert to 0 or 1

    category_name = CATEGORY_NAMES[predicted_category]
    attribute_names = [ATTRIBUTE_NAMES[i + 1] for i, val in enumerate(predicted_attributes[0]) if val == 1]

    return f"{category_name} {' '.join(filter(None, attribute_names))}"