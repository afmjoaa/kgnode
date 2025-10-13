from kgnode._sparql_query import _sparql_query
from typing import List, Dict

def search_entities_by_keywords(keywords: list, limit: int = 10, endpoint_url: str = "http://localhost:7878/query"):
    """
    Search for entities in the knowledge graph based on keywords.

    Args:
        keywords (list): List of keyword strings to search for in entity labels.
        limit (int): Maximum number of results to return. Default is 10.
        endpoint_url (str): URL of the SPARQL endpoint. Default is local Oxigraph.

    Returns:
        List[Dict]: Query results with 'entity' and 'label' keys.
    """
    if not keywords:
        return []

    # Build FILTER conditions for each keyword
    filter_conditions = []
    for keyword in keywords:
        # Escape special characters and convert to lowercase for case-insensitive search
        escaped_keyword = keyword.replace("\\", "\\\\").replace('"', '\\"')
        filter_conditions.append(f'CONTAINS(LCASE(STR(?label)), "{escaped_keyword.lower()}")')

    # Combine all conditions with OR
    filter_clause = " || ".join(filter_conditions)

    # Construct the SPARQL query
    sparql_query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?entity ?label
    WHERE {{
      ?entity rdfs:label ?label .
      FILTER({filter_clause})
    }}
    LIMIT {limit}
    """

    return _sparql_query(sparql_query, endpoint_url)


# def create_entity_description(entity_uri: str) -> str:
#     """
#     Create natural language description for an entity by querying all its triples.
#     Knowledge graph agnostic.
#
#     Args:
#         entity_uri: URI of the entity (without angle brackets)
#
#     Returns:
#         Natural language description with clear subject-predicate-object structure
#     """
#     # Remove angle brackets if present
#     entity_uri = entity_uri.strip()
#     if entity_uri.startswith('<') and entity_uri.endswith('>'):
#         entity_uri = entity_uri[1:-1]
#
#     query = f"""
#     PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#     PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#     SELECT ?predicate ?object
#     WHERE {{
#         <{entity_uri}> ?predicate ?object .
#     }}
#     """
#
#     triples = _sparql_query(query)
#
#     if not triples:
#         return _uri_to_label(entity_uri)
#
#     entity_label = _uri_to_label(entity_uri)
#     description_parts = []
#
#     for triple in triples:
#         predicate = triple['predicate']
#         obj = triple['object']
#
#         pred_label = _uri_to_label(predicate)
#         obj_label = _uri_to_label(obj)
#
#         # Explicit structure for better semantic understanding
#         description_parts.append(
#             f"{entity_label} has {pred_label} {obj_label}"
#         )
#
#     return ". ".join(description_parts)
#
# def create_entity_descriptions_batch(entity_uris: List[str]) -> Dict[str, str]:
#     """
#     Create natural language descriptions for multiple entities in a single query.
#     Much faster than querying entities one by one.
#
#     Args:
#         entity_uris: List of entity URIs to get descriptions for
#
#     Returns:
#         Dictionary mapping entity URI to its description
#     """
#     if not entity_uris:
#         return {}
#
#     # Build VALUES clause for batch query
#     values_clause = " ".join([f"<{uri}>" for uri in entity_uris])
#
#     query = f"""
#     PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#     PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#
#     SELECT ?entity ?predicate ?object
#     WHERE {{
#         VALUES ?entity {{ {values_clause} }}
#         ?entity ?predicate ?object .
#     }}
#     """
#
#     triples = _sparql_query(query)
#
#     # Group triples by entity
#     entity_triples = {}
#     for triple in triples:
#         entity = triple['entity']
#         if entity not in entity_triples:
#             entity_triples[entity] = []
#         entity_triples[entity].append(triple)
#
#     # Create descriptions
#     descriptions = {}
#     for entity_uri in entity_uris:
#         if entity_uri not in entity_triples:
#             # Entity has no triples
#             descriptions[entity_uri] = _uri_to_label(entity_uri)
#         else:
#             entity_label = _uri_to_label(entity_uri)
#             description_parts = []
#
#             for triple in entity_triples[entity_uri]:
#                 predicate = triple['predicate']
#                 obj = triple['object']
#
#                 pred_label = _uri_to_label(predicate)
#                 obj_label = _uri_to_label(obj)
#
#                 description_parts.append(
#                     f"{entity_label} has {pred_label} {obj_label}"
#                 )
#
#             descriptions[entity_uri] = ". ".join(description_parts)
#
#     return descriptions
#

def create_entity_description(entity_uri: str) -> str:
    """
    Create a focused natural language description optimized for semantic search.
    Prioritizes key identifying information: names, titles, venues, years.

    Args:
        entity_uri: URI of the entity (without angle brackets)

    Returns:
        Natural language description optimized for search
    """
    # Remove angle brackets if present
    entity_uri = entity_uri.strip()
    if entity_uri.startswith('<') and entity_uri.endswith('>'):
        entity_uri = entity_uri[1:-1]

    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dblp: <https://dblp.org/rdf/schema#>
    SELECT ?predicate ?object
    WHERE {{
        <{entity_uri}> ?predicate ?object .
    }}
    """

    triples = _sparql_query(query)

    if not triples:
        return _uri_to_label(entity_uri)

    # Organize triples by priority
    entity_type = None
    title = None
    authors = []
    venue = None
    year = None
    affiliation = None
    coauthors = []

    for triple in triples:
        predicate = triple['predicate']
        obj = triple['object']
        pred_label = _uri_to_label(predicate).lower()
        obj_label = _uri_to_label(obj)

        # Identify entity type
        if 'type' in pred_label:
            entity_type = obj_label.lower()

        # Extract key fields based on predicate
        elif 'title' in pred_label:
            title = obj_label
        elif 'authored by' in pred_label or 'author' in pred_label or 'creator' in pred_label:
            authors.append(obj_label)
        elif 'published in' in pred_label or 'venue' in pred_label or 'journal' in pred_label:
            venue = obj_label
        elif 'year' in pred_label:
            year = obj_label
        elif 'affiliation' in pred_label or 'organization' in pred_label:
            affiliation = obj_label
        elif 'coauthor' in pred_label or 'collaborate' in pred_label:
            coauthors.append(obj_label)

    # Get base entity label
    entity_label = _uri_to_label(entity_uri)

    # Build focused description based on entity type
    description_parts = []

    # Always start with the entity label/name
    if entity_type == 'person' or entity_type == 'creator':
        description_parts.append(f"Person: {entity_label}")
        if affiliation:
            description_parts.append(f"affiliated with {affiliation}")
        if authors:  # These are actually papers they authored
            description_parts.append(f"author of {len(authors)} publications")
        if coauthors:
            coauthor_names = ", ".join(coauthors[:5])
            description_parts.append(f"collaborates with {coauthor_names}")

    elif entity_type in ['article', 'publication', 'inproceedings', 'informal']:
        description_parts.append(f"Publication: {entity_label}")
        if title:
            description_parts.append(f"titled '{title}'")
        if authors:
            # Limit to first few authors for readability
            author_names = ", ".join(authors[:10])
            if len(authors) > 10:
                author_names += f" and {len(authors) - 10} more"
            description_parts.append(f"authored by {author_names}")
        if venue:
            description_parts.append(f"published in {venue}")
        if year:
            description_parts.append(f"in year {year}")

    else:
        # Generic fallback for other entity types
        description_parts.append(f"{entity_type or 'Entity'}: {entity_label}")
        if title:
            description_parts.append(f"titled '{title}'")
        if authors:
            description_parts.append(f"associated with authors: {', '.join(authors[:5])}")
        if venue:
            description_parts.append(f"venue: {venue}")
        if year:
            description_parts.append(f"year: {year}")

    # Join with proper punctuation
    return ". ".join(description_parts) + "."


def create_entity_descriptions_batch(entity_uris: List[str]) -> Dict[str, str]:
    """
    Create focused descriptions for multiple entities in batch.
    Optimized for semantic search of author names and paper titles.
    """
    if not entity_uris:
        return {}

    # Clean URIs
    cleaned_uris = []
    for uri in entity_uris:
        uri = uri.strip()
        if not uri:
            continue
        if uri.startswith('<') and uri.endswith('>'):
            uri = uri[1:-1]
        cleaned_uris.append(uri)

    if not cleaned_uris:
        return {}

    # Build VALUES clause
    values_clause = " ".join([f"<{uri}>" for uri in cleaned_uris])

    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dblp: <https://dblp.org/rdf/schema#>

    SELECT ?entity ?predicate ?object
    WHERE {{
        VALUES ?entity {{ {values_clause} }}
        ?entity ?predicate ?object .
    }}
    """

    triples = _sparql_query(query)

    # Group triples by entity
    entity_triples = {}
    for triple in triples:
        entity = triple['entity']
        if entity not in entity_triples:
            entity_triples[entity] = []
        entity_triples[entity].append(triple)

    # Create descriptions using the same logic
    descriptions = {}
    for original_uri, cleaned_uri in zip(entity_uris, cleaned_uris):
        if cleaned_uri not in entity_triples:
            descriptions[original_uri] = _uri_to_label(original_uri)
        else:
            # Use similar logic as create_entity_description
            entity_type = None
            title = None
            authors = []
            venue = None
            year = None
            affiliation = None

            for triple in entity_triples[cleaned_uri]:
                pred_label = _uri_to_label(triple['predicate']).lower()
                obj_label = _uri_to_label(triple['object'])

                if 'type' in pred_label:
                    entity_type = obj_label.lower()
                elif 'title' in pred_label:
                    title = obj_label
                elif 'authored by' in pred_label or 'author' in pred_label or 'creator' in pred_label:
                    authors.append(obj_label)
                elif 'published in' in pred_label or 'venue' in pred_label or 'journal' in pred_label:
                    venue = obj_label
                elif 'year' in pred_label:
                    year = obj_label
                elif 'affiliation' in pred_label:
                    affiliation = obj_label

            entity_label = _uri_to_label(cleaned_uri)
            description_parts = []

            if entity_type == 'person' or entity_type == 'creator':
                description_parts.append(f"Person: {entity_label}")
                if affiliation:
                    description_parts.append(f"affiliated with {affiliation}")
                if len(authors) > 0:
                    description_parts.append(f"author of {len(authors)} publications")

            elif entity_type in ['article', 'publication', 'inproceedings', 'informal']:
                description_parts.append(f"Publication: {entity_label}")
                if title:
                    description_parts.append(f"titled '{title}'")
                if authors:
                    author_names = ", ".join(authors[:10])
                    if len(authors) > 10:
                        author_names += f" and {len(authors) - 10} more"
                    description_parts.append(f"authored by {author_names}")
                if venue:
                    description_parts.append(f"published in {venue}")
                if year:
                    description_parts.append(f"in year {year}")
            else:
                description_parts.append(f"{entity_type or 'Entity'}: {entity_label}")

            descriptions[original_uri] = ". ".join(description_parts) + "." if description_parts else entity_label

    return descriptions

def create_relation_description(relation_uri: str) -> str:
    """
    Create natural language description for a relation.
    Knowledge graph agnostic - converts URI to readable text.

    Args:
        relation_uri: URI of the relation/predicate

    Returns:
        Natural language description
    """
    return _uri_to_label(relation_uri)


def _uri_to_label(uri: str) -> str:
    """
    Convert URI to human-readable label.
    Works for any knowledge graph by extracting and formatting the URI fragment.

    Args:
        uri: Full URI (e.g., "http://example.org/ontology#hasName" or "<http://example.org/ontology#hasName>")

    Returns:
        Human-readable label (e.g., "has name")
    """
    # Remove SPARQL brackets if present
    uri = uri.strip('<>')

    # Extract the fragment/local name from URI
    if '#' in uri:
        label = uri.split('#')[-1]
    elif '/' in uri:
        label = uri.split('/')[-1]
    else:
        label = uri

    # Convert camelCase or PascalCase to spaces
    # hasName -> has Name -> has name
    import re
    label = re.sub(r'([a-z])([A-Z])', r'\1 \2', label)

    # Convert snake_case or kebab-case to spaces
    label = label.replace('_', ' ').replace('-', ' ')

    # Lowercase and clean up
    label = label.lower().strip()

    return label


if __name__ == "__main__":
    # print(create_entity_description("https://dblp.org/rdf/schema#Publication"))
    print(create_entity_descriptions_batch(["<https://dblp.org/rec/journals/nature/RheinbayNAWSTHH20>", "https://dblp.org/rdf/schema#Person"]))
    # print(search_entities_by_keywords(["publication", "author"]))
    # print(create_relation_description("http://www.w3.org/2000/01/rdf-schema#comment"))

    # Example usage:
    # entities = ["http://example.org/entity1", "http://example.org/entity2", "http://example.org/entity3"]
    # descriptions = create_entity_descriptions_batch(entities)
    #
    # for entity, desc in descriptions.items():
    #     print(f"{entity}: {desc}")