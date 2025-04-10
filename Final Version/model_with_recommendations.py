"""
Amazon Recommendation Proof of Concept (POC)

This script:
- Uses a trained model to generate a search keyword based on a test image
- Fetches Amazon product results via a GraphQL API
- Filters the most relevant results using string similarity
- Exports the results as both HTML and JSON files for inspection

Dependencies:
    - config.py: constants and API keys
    - utilities.py: image preprocessing, prediction interpretation, and search utilities
"""

import json
from config import *
from utilities import *

def generate_html(results):
    """
    Generates an HTML table of Amazon product search results.

    Args:
        results (list): List of product dictionaries.

    Returns:
        str: HTML-formatted string.
    """
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
                <td><img src="{product.get('mainImageUrl', '#')}" alt="Product Image"></td>
                <td>{product.get('title', 'N/A')}</td>
                <td>{product.get('brand', 'N/A')}</td>
                <td>{product.get('price', {}).get('display', 'Not Available') if product.get('price') else 'Not Available'}</td>
                <td>{product.get('rating', 'N/A')}</td>
                <td><a href="{product.get('url', '#')}" target="_blank">View Product</a></td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """
    return html

def generate_json(results):
    """
    Converts product results to a JSON-formatted string.

    Args:
        results (list): List of product dictionaries.

    Returns:
        str: Pretty-printed JSON string.
    """
    return json.dumps(results, indent=4)

if __name__ == "__main__":
    # Generate search keyword from model prediction on the test image
    search_keyword = get_search_keyword_from_image_test()

    # Fetch and filter Amazon product results
    all_results = fetch_amazon_products(search_keyword)
    filtered_results = filter_relevant_results(all_results, search_keyword, max_results=MAX_SUGGESTIONS)

    # Export results as HTML and JSON
    html_output = generate_html(filtered_results)
    json_output = generate_json(filtered_results)

    html_filename = "amazon_results.html"
    json_filename = "amazon_results.json"

    with open(html_filename, "w", encoding="utf-8") as file:
        file.write(html_output)

    with open(json_filename, "w", encoding="utf-8") as file:
        file.write(json_output)

    print(f"Files saved: {html_filename} and {json_filename} in current directory.")
