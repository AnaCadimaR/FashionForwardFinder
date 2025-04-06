
import json
from config import *
from utilities import *

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
    
    search_keyword = get_search_keyword_from_image_test()

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
