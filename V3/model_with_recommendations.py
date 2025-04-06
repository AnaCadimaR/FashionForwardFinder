import requests
import json
import Levenshtein
from config import *
from utilities import *


url = "https://graphql.canopyapi.co/"

headers = {
    "Content-Type": "application/json",
    "API-KEY": AMAZON_API_KEY,
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
    <html>
    <head>
        <title>Amazon Product Search Results</title>
        <style>
            table { width: 100%%; border-collapse: collapse; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f4f4f4; }
            img { width: 100px; height: auto; }
        </style>
    </head>
    <body>
        <h2>Amazon Product Search Results</h2>
        <table>
            <tr>
                <th>Image</th>
                <th>Title</th>
                <th>Brand</th>
                <th>Price</th>
                <th>Rating</th>
                <th>Link</th>
            </tr>
    """

    for product in results:
        html += f"""
            <tr>
                <td><img src=\"{product.get('mainImageUrl', '#')}\" alt=\"Product Image\"></td>
                <td>{product.get('title', 'N/A')}</td>
                <td>{product.get('brand', 'N/A')}</td>
                <td>{product.get('price', {}).get('display', 'Not Available') if product.get('price') else 'Not Available'}</td>
                <td>{product.get('rating', 'N/A')}</td>
                <td><a href=\"{product.get('url', '#')}\" target=\"_blank\">View Product</a></td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """
    return html

def generate_json(results):
    return json.dumps(results, indent=4)

if __name__ == "__main__":
    
    search_keyword = get_search_keyword_from_image()

    all_results = fetch_amazon_products(search_keyword)
    filtered_results = filter_relevant_results(all_results, search_keyword, max_results=MAX_SUGGESTIONS)

    html_output = generate_html(filtered_results)
    json_output = generate_json(filtered_results)

    html_filename = "amazon_results.html"
    json_filename = "amazon_results.json"

    with open(html_filename, "w", encoding="utf-8") as file:
        file.write(html_output)

    with open(json_filename, "w", encoding="utf-8") as file:
        file.write(json_output)

    print(f"Files saved: {html_filename} and {json_filename} in current directory.")
