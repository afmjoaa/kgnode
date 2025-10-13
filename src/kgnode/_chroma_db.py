import time

import pandas as pd
from kgnode._templates import create_entity_descriptions_batch

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os


def add_descriptions_to_csv_batched(csv_path: str, output_path: str = None, batch_size: int = 80) -> pd.DataFrame:
    """
    Read CSV and add descriptions using batched queries.

    Args:
        csv_path: Path to input CSV file
        output_path: Path to save output CSV
        batch_size: Number of entities to query at once, query should fit in 8kb, so batch size max 100.

    Returns:
        DataFrame with added 'description' column
    """
    df = pd.read_csv(csv_path)

    if 'entity' not in df.columns:
        raise ValueError("CSV must have an 'entity' column")

    entity_uris = df['entity'].unique().tolist()

    print(f"Processing {len(entity_uris)} unique entities in batches of {batch_size}...")

    # Process in batches
    all_descriptions = {}
    for i in range(0, len(entity_uris), batch_size):
        batch = entity_uris[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(entity_uris) - 1) // batch_size + 1}...")
        batch_descriptions = create_entity_descriptions_batch(batch)
        all_descriptions.update(batch_descriptions)

    # Map descriptions
    df['description'] = df['entity'].map(all_descriptions)
    df['description'] = df['description'].fillna('')

    if output_path is None:
        output_path = csv_path

    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")

    return df


def compile_chromadb(
        csv_path: str,
        collection_name: str = "top_entity_descriptions",
        persist_directory: str = "../../_data/vector_db/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        batch_size: int = 1000,
        force_recreate: bool = False
) -> chromadb.Collection:
    """
    Compile a ChromaDB collection from a CSV file with entity descriptions.
    Uses batch insertion for fast performance.

    Args:
        csv_path: Path to CSV file with 'entity' and 'description' columns
        collection_name: Name for the ChromaDB collection
        persist_directory: Directory to persist the database
        embedding_model: Sentence transformer model name for embeddings
        batch_size: Number of documents to insert per batch (larger = faster but more memory)
        force_recreate: If True, delete existing collection and create new one

    Returns:
        ChromaDB collection object
    """
    # Read the CSV file
    print(f"Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Validate required columns
    if 'entity' not in df.columns or 'description' not in df.columns:
        raise ValueError("CSV must have 'entity' and 'description' columns")

    # Remove rows with empty descriptions
    df = df.dropna(subset=['description'])
    df = df[df['description'].str.strip() != '']

    print(f"Found {len(df)} entities with descriptions")

    # Initialize ChromaDB client with persistence
    print(f"Initializing ChromaDB at {persist_directory}...")
    client = chromadb.PersistentClient(path=persist_directory)

    # Delete existing collection if force_recreate is True
    if force_recreate:
        try:
            client.delete_collection(name=collection_name)
            print(f"Deleted existing collection '{collection_name}'")
        except:
            pass

    # Create or get collection with the specified embedding model
    print(f"Creating collection '{collection_name}' with model '{embedding_model}'...")
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "description": "Entity descriptions from knowledge graph"
        },
        embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
    )

    # Check if collection already has data
    existing_count = collection.count()
    if existing_count > 0 and not force_recreate:
        print(f"Collection already has {existing_count} documents. Use force_recreate=True to rebuild.")
        return collection

    # Prepare data for batch insertion
    entities = df['entity'].tolist()
    descriptions = df['description'].tolist()

    # Insert in batches for performance
    total_batches = (len(entities) + batch_size - 1) // batch_size
    print(f"\nInserting {len(entities)} entities in {total_batches} batches...")

    for i in range(0, len(entities), batch_size):
        batch_entities = entities[i:i + batch_size]
        batch_descriptions = descriptions[i:i + batch_size]

        # Use entity URIs as IDs (ChromaDB requires string IDs)
        # If URIs are too long or have special chars, create simpler IDs
        batch_ids = [f"entity_{j}" for j in range(i, min(i + batch_size, len(entities)))]

        # Store original entity URI in metadata
        batch_metadatas = [{"entity_uri": entity} for entity in batch_entities]

        collection.add(
            ids=batch_ids,
            documents=batch_descriptions,
            metadatas=batch_metadatas
        )

        batch_num = (i // batch_size) + 1
        print(f"Batch {batch_num}/{total_batches} completed ({len(batch_entities)} entities)")

    final_count = collection.count()
    print(f"\n✓ ChromaDB compilation complete!")
    print(f"✓ Total entities indexed: {final_count}")
    print(f"✓ Collection: {collection_name}")
    print(f"✓ Persisted at: {persist_directory}")

    return collection


def search_similar_entities(
        collection: chromadb.Collection,
        query: str,
        n_results: int = 5
) -> List[Dict[str, any]]:
    """
    Search for similar entities based on query description.

    Args:
        collection: ChromaDB collection
        query: Query text to search for
        n_results: Number of results to return

    Returns:
        List of dictionaries with entity URIs, descriptions, and similarity scores
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    # Format results
    formatted_results = []
    for i in range(len(results['ids'][0])):
        formatted_results.append({
            'entity_uri': results['metadatas'][0][i]['entity_uri'],
            'description': results['documents'][0][i],
            'distance': results['distances'][0][i] if 'distances' in results else None,
            'id': results['ids'][0][i]
        })

    return formatted_results


def load_chromadb(
        collection_name: str = "top_entity_descriptions",
        persist_directory: str = "../../_data/vector_db/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
) -> chromadb.Collection:
    """
    Load an existing ChromaDB collection.

    Args:
        collection_name: Name of the collection
        persist_directory: Directory where database is persisted
        embedding_model: Embedding model name (must match the one used during creation)

    Returns:
        ChromaDB collection object
    """
    client = chromadb.PersistentClient(path=persist_directory)

    collection = client.get_collection(
        name=collection_name,
        embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
    )

    print(f"Loaded collection '{collection_name}' with {collection.count()} entities")
    return collection


if __name__ == "__main__":
    # Add description to csv speed checking
    # start_time = time.time()
    #
    # add_descriptions_to_csv_batched("./test_top_entities.csv", batch_size=80)
    #
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    #
    # print(f"\n{'=' * 60}")
    # print(f"Total execution time: {elapsed_time:.2f} seconds")
    # print(f"Total execution time: {elapsed_time / 60:.2f} minutes")
    # print(f"{'=' * 60}")

    # ==================================================================
    # Compile ChromaDB from CSV (first time or recreate)
    start_time = time.time()

    collection = compile_chromadb(
        csv_path="./test_top_entities.csv",
        embedding_model="all-MiniLM-L6-v2",
        batch_size=1000,  # Adjust based on your memory
        force_recreate=True  # Set to False to skip if already exists
    )

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds")

    # Search for similar entities
    print("\n" + "=" * 60)
    print("TESTING SEARCH")
    print("=" * 60)

    query = "somatic drivers in 2,658 cancer whole genomes"
    results = search_similar_entities(collection, query, n_results=3)

    print(f"\nQuery: {query}\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Entity: {result['entity_uri']}")
        print(f"   Distance: {result['distance']:.4f}")
        print(f"   Description: {result['description'][:200]}...")
        print()