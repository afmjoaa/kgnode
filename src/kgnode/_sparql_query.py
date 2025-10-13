from SPARQLWrapper import SPARQLWrapper, JSON


def _sparql_query(sparql_query: str, endpoint_url="http://localhost:7878/query"):
    """
    Query a running Oxigraph SPARQL endpoint and return results as a list of dictionaries.

    Args:
        sparql_query (str): The SPARQL query string.
        endpoint_url (str): URL of the SPARQL endpoint. Default is local Oxigraph.

    Returns:
        List[Dict]: Query results. Each dict maps variable names to their values.
    """
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)

    sparql_results = sparql.query().convert()

    # Convert results to list of dicts for easier processing
    output = []
    for result in sparql_results["results"]["bindings"]:
        row = {var: result[var]["value"] for var in result}
        output.append(row)

    return output


# Example usage
if __name__ == "__main__":
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT * WHERE {
        ?sub ?pred ?obj .
    } LIMIT 10
    """

    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?predicate ?object
    WHERE {{
        <https://dblp.org/rdf/schema#Publication> ?predicate ?object .
    }}
    """

    keyword_query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?entity ?label
    WHERE {
      ?entity rdfs:label ?label .
      FILTER(CONTAINS(LCASE(STR(?label)), "publication") || CONTAINS(LCASE(STR(?label)), "author"))
    }
    LIMIT 10
    """
    results = _sparql_query(keyword_query)
    print(results)
    for r in results:
        print(r)
