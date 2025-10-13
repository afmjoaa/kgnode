import csv
from functools import lru_cache
from typing import List, Dict
from tqdm import tqdm
import time
import threading

from kgnode._sparql_query import _sparql_query

@lru_cache(maxsize=10)
def get_top_entities_by_degree(limit: int = 1_000_000, output_file: str = "../../_data/vector_db/top_entities_by_degree.csv") -> List[
    Dict[str, str]]:
    """
    Get top N entities from knowledge graph sorted by degree (number of connections).
    Saves results to CSV file.

    Args:
        limit (int): Number of top entities to retrieve. Default is 1,000,000.
        output_file (str): Path to output CSV file. Default is "top_entities_by_degree.csv".

    Returns:
        List[Dict]: List of entities with their URIs and degrees.
                   Each dict has keys: 'entity', 'degree'
    """
    print(f"Querying top {limit:,} entities by degree, For 1 million nodes KG it takes 7 seconds to run.")

    sparql_query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?entity (COUNT(?connection) as ?degree)
    WHERE {{
      {{ ?entity ?p ?o }}
      UNION
      {{ ?s ?p ?entity }}
    }}
    GROUP BY ?entity
    ORDER BY DESC(?degree)
    LIMIT {limit}
    """

    # in+out
    # SELECT ?entity (COUNT(?connection) as ?degree)
    #     WHERE {{
    #       {{ ?entity ?p ?o }}
    #       UNION
    #       {{ ?s ?p ?entity }}
    #     }}
    #     GROUP BY ?entity
    #     ORDER BY DESC(?degree)
    #     LIMIT {limit}

    # indegree
    # SELECT ?entity (COUNT(?s) as ?degree)
    # WHERE {{
    #   ?s ?p ?entity .
    # }}
    # GROUP BY ?entity
    # ORDER BY DESC(?degree)
    # LIMIT {limit}

    # outdegree
    # SELECT ?entity (COUNT(?o) as ?degree)
    #     WHERE {{
    #       ?entity ?p ?o .
    #     }}
    #     GROUP BY ?entity
    #     ORDER BY DESC(?degree)
    #     LIMIT {limit}

    # Spinner setup
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    spinner_running = True

    def spin():
        i = 0
        start = time.time()
        while spinner_running:
            elapsed = int(time.time() - start)
            mins, secs = divmod(elapsed, 60)
            print(f'\r{spinner_chars[i % len(spinner_chars)]} Querying... {mins:02d}:{secs:02d}', end='', flush=True)
            i += 1
            time.sleep(0.1)

    # Start spinner
    spinner_thread = threading.Thread(target=spin)
    spinner_thread.start()

    start_time = time.time()
    results = _sparql_query(sparql_query)
    query_time = time.time() - start_time

    # Stop spinner
    spinner_running = False
    spinner_thread.join()

    print(f"\r✓ Query completed in {query_time:.1f} seconds ({query_time / 60:.1f} minutes)")
    print(f"✓ Retrieved {len(results):,} entities")
    print(f"Saving to {output_file}...")

    # Save to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['entity', 'degree']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(results):
            writer.writerow({
                'entity': row['entity'],
                'degree': row['degree']
            })

            if (idx + 1) % 100_000 == 0:
                print(f"  Written {idx + 1:,} rows...")

    total_time = time.time() - start_time
    print(f"✓ Done! Saved {len(results):,} entities to {output_file}")
    print(f"Total time: {total_time:.1f} seconds")

    return results


if __name__ == "__main__":
    entities = get_top_entities_by_degree(limit=10000)
    print(entities)