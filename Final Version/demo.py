from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from config import MAX_SUGGESTIONS
from utilities import *

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
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
