import tensorflow as tf
from config import MAX_HEIGHT, MAX_WIDTH

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


def preprocess_single_image(img_path):
    """Loads and preprocesses a single image for inference"""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3) 
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_pad(img, MAX_HEIGHT, MAX_WIDTH) 
    return img
