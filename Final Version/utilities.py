"""
Utility functions for preprocessing images, generating search keywords using model predictions,
and fetching relevant product results from Amazon based on those predictions.

This module supports:
- Preprocessing training and test images
- Generating category and attribute labels from model outputs
- Constructing a keyword string for Amazon product search
- Querying and filtering relevant product results using semantic similarity

Dependencies:
    - TensorFlow
    - NumPy
    - Requests
    - Levenshtein
    - Custom config and query modules
"""

import requests
import numpy as np
import tensorflow as tf
import Levenshtein
from tensorflow.keras.models import load_model
from config import *
from amazon_queries import *

def preprocess_image(img_path, category_label, attribute_label):
    """
    Loads and preprocesses an image and its labels for training.

    Args:
        img_path (str): File path to the image.
        category_label (int): Category label.
        attribute_label (array-like): Multi-label attribute vector.

    Returns:
        Tuple: (processed image tensor, (category label, attribute label))
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_pad(img, MAX_HEIGHT, MAX_WIDTH)

    return img, (tf.cast(category_label, tf.int32), tf.cast(attribute_label, tf.float32))

def preprocess_test_image(img_path):
    """
    Preprocesses a test image for inference.

    Args:
        img_path (str): File path to the image.

    Returns:
        Tensor: A batch of one preprocessed image.
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3) 
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_pad(img, MAX_HEIGHT, MAX_WIDTH) 
    img = tf.expand_dims(img, axis=0)  
    return img

def get_search_keyword_from_image_test() -> str:
    """
    Predicts the category and attributes of a predefined test image and builds a search keyword.

    Returns:
        str: Search keyword composed of category and applicable attributes.
    """
    model = load_model(MODEL_SAVE_PATH)
    test_image = preprocess_test_image(TEST_IMAGE_PATH)
    cat, att = model.predict(test_image)

    predicted_category = convert_to_category_output(cat)
    predicted_attributes = convert_to_attribute_output(att) 

    category_info = CATEGORY_NAMES[predicted_category]
    category_name = category_info['desc']
    cat_type = category_info['cat_type']
    attribute_descriptions = get_applicable_attributes(predicted_attributes, cat_type)

    return f"{category_name} {' '.join(attribute_descriptions)}"

def get_search_keyword_from_image(image_url: str) -> str:
    """
    Predicts the category and attributes of an image from a given path and builds a search keyword.

    Args:
        image_url (str): Path to the image.

    Returns:
        str: Search keyword composed of category and applicable attributes.
    """
    model = load_model(MODEL_SAVE_PATH)
    test_image = preprocess_test_image(image_url)
    cat, att = model.predict(test_image)

    predicted_category = convert_to_category_output(cat)
    predicted_attributes = convert_to_attribute_output(att) 

    category_info = CATEGORY_NAMES[predicted_category]
    category_name = category_info['desc']
    cat_type = category_info['cat_type']
    attribute_descriptions = get_applicable_attributes(predicted_attributes, cat_type)

    return f"{category_name} {' '.join(attribute_descriptions)}"

def get_search_keyword_from_prediction(predicted_category, predicted_attributes) -> str:
    """
    Builds a search keyword from predicted category and attribute vectors.

    Args:
        predicted_category (int): Predicted category index.
        predicted_attributes (array-like): Binary vector of predicted attributes.

    Returns:
        str: Combined keyword string.
    """
    category_info = CATEGORY_NAMES[predicted_category]
    category_name = category_info['desc']
    cat_type = category_info['cat_type']
    attribute_descriptions = get_applicable_attributes(predicted_attributes, cat_type)

    return f"{category_name} {' '.join(attribute_descriptions)}"

def convert_to_category_output(predicted_category) -> int:
    """
    Converts the predicted category output (probabilities) to a category index.

    Args:
        predicted_category (array-like): Model output for category.

    Returns:
        int: Predicted category index (1-based).
    """
    predicted_index = np.argmax(predicted_category, axis=1)[0]  
    return predicted_index + 1

def convert_to_attribute_output(predicted_attributes) -> list:
    """
    Converts the predicted attribute probabilities to binary flags.

    Args:
        predicted_attributes (array-like): Model output for attributes.

    Returns:
        list: Binary list of predicted attributes.
    """
    return (predicted_attributes > ATTRIBUTES_SENSIBILITY).astype(int) 

def get_applicable_attributes(attribute_pred, cat_type: int) -> list[str]:
    """
    Filters applicable attributes based on the predicted attribute flags and category type.

    Args:
        attribute_pred (array-like): Binary attribute vector.
        cat_type (int): Category type index.

    Returns:
        list[str]: List of attribute descriptions.
    """
    predicted_flags = (attribute_pred > ATTRIBUTES_SENSIBILITY).astype(int)[0]
    applicable = []
    for i, flag in enumerate(predicted_flags):
        if flag == 1:
            attr = ATTRIBUTE_NAMES[i + 1]
            if not attr['apply_to'] or cat_type in attr['apply_to']:
                applicable.append(attr['desc'])
    return applicable

def fetch_amazon_products(search_keyword):
    """
    Queries the Amazon GraphQL API using a search keyword to fetch product results.

    Args:
        search_keyword (str): Keyword to search products for.

    Returns:
        list[dict]: List of product results, each as a dictionary.
    """
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
    """
    Computes a normalized Levenshtein similarity score between a product title and keyword.

    Args:
        title (str): Product title.
        search_keyword (str): Search keyword.

    Returns:
        float: Similarity score in range [0, 1], where 1 is an exact match.
    """
    return 1 - (Levenshtein.distance(title.lower(), search_keyword.lower()) / max(len(title), len(search_keyword)))

def filter_relevant_results(results, search_keyword, max_results=10, threshold=0.4):
    """
    Filters and ranks the most relevant product results based on title similarity.

    Args:
        results (list[dict]): List of product dictionaries.
        search_keyword (str): Keyword to compare against.
        max_results (int): Maximum number of results to return.
        threshold (float): Minimum similarity threshold.

    Returns:
        list[dict]: Top-N relevant product results.
    """
    scored_results = [(product, calculate_similarity(product.get("title", ""), search_keyword)) for product in results]
    scored_results.sort(key=lambda x: x[1], reverse=True)

    relevant_results = [item[0] for item in scored_results if item[1] >= threshold]

    if len(relevant_results) < max_results:
        relevant_results = [item[0] for item in scored_results[:max_results]]

    return relevant_results[:max_results]
