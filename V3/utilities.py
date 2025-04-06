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
    """Extracts search keyword from the image filename using model prediction."""

    model = load_model(MODEL_SAVE_PATH)
    test_image = preprocess_test_image(TEST_IMAGE_PATH)
    cat, att = model.predict(test_image)

    predicted_category =  convert_to_category_output(cat)
    predicted_attributes = convert_to_attribute_output(att) 
    
    category_info = CATEGORY_NAMES[predicted_category]
    
    category_name = category_info['desc']
    cat_type = category_info['cat_type']
    attribute_descriptions = get_applicable_attributes(predicted_attributes, cat_type)

    return f"{category_name} {' '.join(attribute_descriptions)}"

def get_search_keyword_from_prediction(predicted_category, predicted_attributes) -> str:
    category_info = CATEGORY_NAMES[predicted_category]
    
    category_name = category_info['desc']
    cat_type = category_info['cat_type']
    attribute_descriptions = get_applicable_attributes(predicted_attributes, cat_type)

    return f"{category_name} {' '.join(attribute_descriptions)}"


def convert_to_category_output(predicted_category) -> int:
    """Converts predicted category index to actual category."""
    predicted_index =  np.argmax(predicted_category, axis=1)[0]  
    return predicted_index + 1

def convert_to_attribute_output(predicted_attributes) -> list:
    """Converts predicted attributes to a list of the actual attributes"""
    return (predicted_attributes > ATTRIBUTES_SENSIBILITY).astype(int) 

def get_applicable_attributes(attribute_pred, cat_type: int) -> list[str]:
    predicted_flags = (attribute_pred > ATTRIBUTES_SENSIBILITY).astype(int)[0]
    applicable = []
    for i, flag in enumerate(predicted_flags):
        if flag == 1:
            attr = ATTRIBUTE_NAMES[i + 1]
            if not attr['apply_to'] or cat_type in attr['apply_to']:
                applicable.append(attr['desc'])
    return applicable