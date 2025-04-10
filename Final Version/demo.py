"""
Proof of Concept #2: Flask Web Application for Fashion Item Search

This app allows users to upload an image of a fashion item, uses a trained model to extract
category and attribute information, generates a search keyword, queries Amazon via GraphQL,
and displays relevant product matches.

Core Features:
- Image upload via web UI
- Prediction using deep learning model
- Keyword generation based on predicted labels
- Amazon product fetching and filtering
- HTML rendering via Jinja templates

To run:
    python app.py
Then navigate to http://localhost:5000 in your browser.

Dependencies:
    - Flask
    - TensorFlow
    - Utilities and config files from this project
"""

from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from config import MAX_SUGGESTIONS
from utilities import *

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """
    Checks if the uploaded file has a supported image extension.

    Args:
        filename (str): Name of the file to check.

    Returns:
        bool: True if file has a valid extension, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    """
    Handles the main web page for uploading an image and displaying product results.

    GET:
        - Renders the file upload form (index.html)

    POST:
        - Validates and saves the uploaded file
        - Runs model inference to generate keyword
        - Fetches and filters Amazon product results
        - Renders results page (results.html)

    Returns:
        HTML: Rendered template (index or results)
    """
    results = []
    keyword = ""

    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part"
        file = request.files['image']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            keyword = get_search_keyword_from_image(filepath)
            all_results = fetch_amazon_products(keyword)
            results = filter_relevant_results(all_results, keyword, max_results=MAX_SUGGESTIONS)

            return render_template('results.html', results=results, keyword=keyword)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
