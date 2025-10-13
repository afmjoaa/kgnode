# kgnode
Knowledge Graph Agnostic Node for Knowledge-Aware LLM Applications


## Classes

### 1. `_sparql_query.py`
This module provides the core function for communicating with SPARQL endpoints.

#### Function: `_sparql_query`
* **Signature:** `(sparql_query: str, endpoint_url: str) -> List[Dict]`
* **Description:** Executes a given **SPARQL query** against the specified **endpoint URL**. It returns the results as a list of dictionaries

### 2. _vector_db.py
Create a VectorDB given the vector store type knowledge graph config and seed nodes with the following functionalities:

#### Function: `_compile`
* **Signature:** `----`
* **Description:** Add knowledge to the knowledge graph and vector database.


      - `add_or_update_vectors`: Add entities embedding to the vectordb.
      - `search_nodes`: Search for the seed node using symentic search.
      - `delete_vectors`: Delete entities embedding from the vectordb.

  
### 3. _kg_node.py
The main entry point for the knowledge graph node with the following functionalities:
        - `citable`: Lightweight function to check if seed node is good enough or not.
        - `get_seed_node`: Return the seed node using sparql keyword search and vectordb semantic search
        - `get_subgraph`: Get the relevant subgraphs for answering the query 
        - `get_sparql`: Get the sparql using the relevant subgraph and user query
        - `retrieve`: Retrieve the context from the knowledge graph to answer the query in natural language
        - `generate_answer`: Generate appropriate answer using the retrieved context.
        - `print_architecture`: Print the architecture both langgraph and workflow

### VectorDBs
1. Chomadb
2. Pinecone
3. Qdrant

### Knowledge graph embedding models - Nope


### Test embedding models with predefined schema - Yes
    1. Max two model - literal
        - all-MiniLM-L6-v2
        - google/embeddinggemma-300m


### Dataset : https://dblp.org/rdf/
    DBLP-QuAD == 252 million triples : https://zenodo.org/records/7638511
    91626965 == 92 million entities
    62 relationship
    paper link: https://www.inf.uni-hamburg.de/en/inst/ab/lt/publications/2023-banerjee-bir-ecir-2023-dblpquad.pdf


### Local KG server
    oxigraph_server

run: oxigraph_server -l ~/oxigraph_db --bind 127.0.0.1:7878
load: oxigraph_server load -l ./oxigraph_db -f dblp.nt
serve (read): oxigraph_server serve-read-only -l ./oxigraph_db --cors 
serve (RW): oxigraph_server serve -l ./oxigraph_db --cors 


#### Keyword based search 
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?entity ?label
WHERE {
  ?entity rdfs:label ?label .
  FILTER(CONTAINS(LCASE(STR(?label)), "publication") || CONTAINS(LCASE(STR(?label)), "author"))
}
LIMIT 10
