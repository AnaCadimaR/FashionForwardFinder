import os
import numpy as np
import tensorflow as tf
import requests
import json
import webbrowser
import Levenshtein
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from config import BASE_PATH, BATCH_SIZE, EPOCHS, MODEL_SAVE_PATH
from utilities import preprocess_image
from utilities import preprocess_test_image
from tensorflow.keras.models import load_model
from utilities import preprocess_single_image


#Loading category --corregir valores del diccionario--
category_names = {
    0:'Anorak',1:'Blazer',2:'Blouse',3:'Bomber',4:'Button-Down',5:'Cardigan',6:'Flannel',7:'Halter',8:'Henley',9:'Hoodie',
    10:'Jacket',11:'Jersey',12:'Parka',13:'Peacoat',14:'Poncho',15:'Sweater',16:'Tank',17:'Tee',18:'Top',19:'Turtleneck',20:'Capris',
    21:'Chinos',22:'Culottes',23:'Cutoffs',24:'Gauchos',25:'Jeans',26:'Jeggings',27:'Jodhpurs',28:'Joggers',29:'Leggings',30:'Sarong',
    31:'Shorts',32:'Skirt',33:'Sweatpants',34:'Sweatshorts',35:'Trunks',36:'Caftan',37:'Cape',38:'Coat',39:'Coverup',40:'Dress',
    41:'Jumpsuit',42:'Kaftan',43:'Kimono',44:'Nightdress',45:'Onesie',46:'Robe',47:'Romper',48:'Shirtdress',49:'Sundress'}


#Use that predicted category name as search keyword for Amazon product search
search_keyword = 'Columbia shoes'

#Fetching from API and Filter Results
url = "https://graphql.canopyapi.co/"
headers = {
    "Content-Type": "application/json",
    "API-KEY":"eeda37d1-b608-408c-aafd-9f9d48a63d64",
}
query = """
    query amazonProduct($searchKeyWord: String!) {
        amazonProductSearchResults(
            input: {
                searchTerm: $searchKeyWord,
                domain: CA
            }) {
            productResults {
                results {
                    title
                    brand
                    url
                    isNew
                    price {
                        display
                    }
                    rating
                    mainImageUrl
                }
            }
        }
    }
"""

def fetch_amazon_products(search_keyword):
    variables = {"searchKeyWord": search_keyword}
    payload = {"query": query, "variables": variables}
    response = requests.post(url, headers=headers, json=payload)
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

def generate_html(results):
    if not results:
        return "<p>No relevant products found.</p>"
    html = """
    <html><head>
        <title>Amazon Product Search Results</title>
        <style>
            table { width: 100%%; border-collapse: collapse; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f4f4f4; }
            img { width: 100px; height: auto; }
        </style>
    </head><body>
        <h2>Amazon Product Search Results for {}</h2>
        <table><tr>
            <th>Image</th>
            <th>Title</th>
            <th>Brand</th>
            <th>Price</th>
            <th>Rating</th>
            <th>Link</th>
        </tr>
    """.format(search_keyword)

    for product in results:
        html += f"""
            <tr>
                <td><img src="{product.get('mainImageUrl', '#')}" alt="Product Image"></td>
                <td>{product.get('title', 'N/A')}</td>
                <td>{product.get('brand', 'N/A')}</td>
                <td>{product.get('price', {}).get('display', 'Not Available') if product.get('price') else 'Not Available'}</td>
                <td>{product.get('rating', 'N/A')}</td>
                <td><a href="{product.get('url', '#')}" target="_blank">View Product</a></td>
            </tr>
        """
    html += "</table></body></html>"
    return html

#Fetch, filter, and generate recommendations
all_results = fetch_amazon_products(search_keyword)
filtered_results = filter_relevant_results(all_results, search_keyword)

html_output = generate_html(filtered_results)
json_output = json.dumps(filtered_results, indent=4)

#Files
html_filename = "recommendations.html"
json_filename = "recommendations.json"

with open(html_filename, "w", encoding="utf-8") as file:
    file.write(html_output)

with open(json_filename, "w", encoding="utf-8") as file:
    file.write(json_output)

#To open HTML
webbrowser.open('file://' + os.path.realpath(html_filename))

#To open JSON
os.startfile(os.path.realpath(json_filename))

print(f"Recommendations generated and downloaded for: {search_keyword}")