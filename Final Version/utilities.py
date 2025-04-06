import requests
import numpy as np
import tensorflow as tf
import Levenshtein
from tensorflow.keras.models import load_model
from config import *
from amazon_queries import *

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

def get_search_keyword_from_image_test() -> str:
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

def get_search_keyword_from_image(image_url: str) -> str:
    """Extracts search keyword from the image filename using model prediction."""

    model = load_model(MODEL_SAVE_PATH)
    test_image = preprocess_test_image(image_url)
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

def fetch_amazon_products(search_keyword):
    variables = {"searchKeyWord": search_keyword}
    payload = {"query": GET_PRODUCTS_QUERY, "variables": variables}

    response = requests.post(CANOPY_URL, headers=CANOPY_HEADERS, json=payload)

    if response.status_code == 200:
        data = response.json()
        return data.get("data", {}).get("amazonProductSearchResults", {}).get("productResults", {}).get("results", [])
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []
    
def calculate_similarity(title, search_keyword):
    return 1 - (Levenshtein.distance(title.lower(), search_keyword.lower()) / max(len(title), len(search_keyword)))

def filter_relevant_results(results, search_keyword, max_results=10, threshold=0.4):
    scored_results = [(product, calculate_similarity(product.get("title", ""), search_keyword)) for product in results]
    scored_results.sort(key=lambda x: x[1], reverse=True)

    relevant_results = [item[0] for item in scored_results if item[1] >= threshold]

    if len(relevant_results) < max_results:
        relevant_results = [item[0] for item in scored_results[:max_results]]

    return relevant_results[:max_results]