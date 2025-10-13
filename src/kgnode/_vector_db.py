import logging

logger = logging.getLogger(__name__)

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple


class VectorDB:
    def __init__(
            self,
            vector_store_type: str,
            kg_config: dict,
            embedding_model_name: str = "all-MiniLM-L6-v2",
            seed_nodes: list = None,
            **kwargs
    ):
        """
        Initialize VectorDB with path-aware capabilities.

        Args:
            vector_store_type: Type of vector store
            kg_config: Knowledge graph configuration
            embedding_model_name: Name of the sentence transformer model
            seed_nodes: Initial seed nodes
        """
        self.vector_store_type = vector_store_type
        self.kg_config = kg_config

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Storage for embeddings
        self.entity_embeddings = {}  # {entity_uri: embedding_vector}
        self.relation_embeddings = {}  # {relation_uri: embedding_vector}

        self.is_compiled = False
        self.node_count = 0

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Create embedding for a text using the sentence transformer model.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array
        """
        return self.embedding_model.encode(text, convert_to_numpy=True)

    def get_entity_embedding(self, entity_uri: str) -> np.ndarray:
        """
        Get or create embedding for an entity.

        Args:
            entity_uri: URI of the entity

        Returns:
            Embedding vector
        """
        if entity_uri not in self.entity_embeddings:
            # Create description and embed
            description = self.create_entity_description(entity_uri)
            self.entity_embeddings[entity_uri] = self._embed_text(description)

        return self.entity_embeddings[entity_uri]

    def get_relation_embedding(self, relation_uri: str) -> np.ndarray:
        """
        Get or create embedding for a relation.

        Args:
            relation_uri: URI of the relation

        Returns:
            Embedding vector
        """
        if relation_uri not in self.relation_embeddings:
            # Create description and embed
            description = self.create_relation_description(relation_uri)
            self.relation_embeddings[relation_uri] = self._embed_text(description)

        return self.relation_embeddings[relation_uri]

    def create_path_embedding(self, path: List[Tuple[str, str, str]]) -> np.ndarray:
        """
        Create path embedding using sequential concatenation + mean pooling.

        Path format: [(subject1, relation1, object1), (subject2, relation2, object2), ...]
        Example: [("BillGates", "founded", "Microsoft"), ("Microsoft", "locatedIn", "Seattle")]

        Args:
            path: List of triples (subject_uri, relation_uri, object_uri)

        Returns:
            Path embedding vector (same dimension as entity/relation embeddings)
        """
        if not path:
            raise ValueError("Path cannot be empty")

        # Collect all embeddings in sequence
        embedding_sequence = []

        for subject_uri, relation_uri, object_uri in path:
            # Get embeddings for this triple
            subj_emb = self.get_entity_embedding(subject_uri)
            rel_emb = self.get_relation_embedding(relation_uri)
            obj_emb = self.get_entity_embedding(object_uri)

            # Add to sequence in order: subject -> relation -> object
            embedding_sequence.append(subj_emb)
            embedding_sequence.append(rel_emb)
            embedding_sequence.append(obj_emb)

        # Convert to numpy array for efficient computation
        embedding_sequence = np.array(embedding_sequence)  # Shape: (seq_len, embedding_dim)

        # Mean pooling: aggregate into single vector
        path_embedding = np.mean(embedding_sequence, axis=0)  # Shape: (embedding_dim,)

        return path_embedding

    def create_path_embedding_incremental(
            self,
            current_path_embedding: np.ndarray,
            current_path_length: int,
            new_triple: Tuple[str, str, str]
    ) -> np.ndarray:
        """
        Incrementally update path embedding when adding a new hop.
        More efficient for pruning algorithm that grows paths step-by-step.

        Args:
            current_path_embedding: Existing path embedding (mean of all previous embeddings)
            current_path_length: Number of embeddings in current path (not triples, but total emb count)
            new_triple: New triple to add (subject_uri, relation_uri, object_uri)

        Returns:
            Updated path embedding
        """
        subject_uri, relation_uri, object_uri = new_triple

        # Get embeddings for new triple
        subj_emb = self.get_entity_embedding(subject_uri)
        rel_emb = self.get_relation_embedding(relation_uri)
        obj_emb = self.get_entity_embedding(object_uri)

        # Stack new embeddings
        new_embeddings = np.array([subj_emb, rel_emb, obj_emb])  # Shape: (3, embedding_dim)

        # Incremental mean update formula:
        # new_mean = (old_mean * old_count + sum(new_values)) / (old_count + new_count)

        total_new_embeddings = 3  # subject + relation + object
        new_count = current_path_length + total_new_embeddings

        updated_embedding = (
                                    current_path_embedding * current_path_length + np.sum(new_embeddings, axis=0)
                            ) / new_count

        return updated_embedding


    def compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Cosine similarity score (between -1 and 1)
        """
        # Normalize vectors
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)

        # Dot product of normalized vectors = cosine similarity
        return np.dot(emb1_norm, emb2_norm)