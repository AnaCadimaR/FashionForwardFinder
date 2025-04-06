GET_PRODUCTS_QUERY = """
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