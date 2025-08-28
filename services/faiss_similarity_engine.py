import logging
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSSimilarityEngine:
    """FAISS-based similarity engine for efficient nearest neighbor search"""

    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim
        self.indices = {}  # Store separate indices for each feature type
        self.embeddings = {}  # Store embeddings for each feature type
        self.person_count = 0

    def build_indices(self, feature_embeddings: Dict[str, np.ndarray]) -> None:
        """Build FAISS indices for each feature type"""
        logger.info("🔍 Building FAISS indices for efficient similarity search...")

        self.person_count = len(next(iter(feature_embeddings.values())))

        for feature_name, embeddings in feature_embeddings.items():
            logger.info(f"  Building index for {feature_name}: {embeddings.shape}")

            # Ensure embeddings are float32 (FAISS requirement)
            embeddings_f32 = embeddings.astype(np.float32)

            # Normalize embeddings for cosine similarity (using inner product on normalized vectors)
            faiss.normalize_L2(embeddings_f32)

            # Create exact index (IndexFlatIP for inner product = cosine similarity on normalized vectors)
            index = faiss.IndexFlatIP(self.embedding_dim)

            # Add embeddings to index
            index.add(embeddings_f32)

            # Store index and embeddings
            self.indices[feature_name] = index
            self.embeddings[feature_name] = embeddings_f32

        logger.info(f"✅ Built FAISS indices for {len(self.indices)} feature types")

    def find_top_k_similar(
        self, feature_name: str, person_index: int, k: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find top-k most similar people for a given person and feature"""
        if feature_name not in self.indices:
            raise ValueError(f"No index built for feature: {feature_name}")

        # Get the person's embedding
        query_embedding = self.embeddings[feature_name][person_index : person_index + 1]

        # Search for top-k similar (including self)
        similarities, indices = self.indices[feature_name].search(
            query_embedding, k + 1
        )

        # Remove self from results (first result is always self with similarity 1.0)
        mask = indices[0] != person_index
        filtered_indices = indices[0][mask][:k]
        filtered_similarities = similarities[0][mask][:k]

        return filtered_similarities, filtered_indices

    def get_similarity_matrix_sparse(
        self, feature_name: str, top_k: int = 50
    ) -> Dict[Tuple[int, int], float]:
        """Build sparse similarity matrix using FAISS top-k search"""
        if feature_name not in self.indices:
            raise ValueError(f"No index built for feature: {feature_name}")

        logger.info(
            f"🔍 Building sparse similarity matrix for {feature_name} (top-{top_k} per person)"
        )

        sparse_similarities = {}

        for person_i in range(self.person_count):
            # Find top-k most similar people for this person
            similarities, similar_indices = self.find_top_k_similar(
                feature_name, person_i, top_k
            )

            # Store similarities in sparse format
            for sim_score, person_j in zip(similarities, similar_indices):
                # Store both directions for undirected graph
                pair_key = (min(person_i, person_j), max(person_i, person_j))

                # Keep the higher similarity if we've seen this pair before
                if (
                    pair_key not in sparse_similarities
                    or sim_score > sparse_similarities[pair_key]
                ):
                    sparse_similarities[pair_key] = float(sim_score)

        logger.info(
            f"  Generated {len(sparse_similarities)} similarity pairs (vs {self.person_count * (self.person_count - 1) // 2} total possible)"
        )

        return sparse_similarities

    def build_all_sparse_matrices(
        self, top_k: int = 50
    ) -> Dict[str, Dict[Tuple[int, int], float]]:
        """Build sparse similarity matrices for all features"""
        logger.info(
            f"🚀 Building sparse similarity matrices (top-{top_k} per person)..."
        )

        all_sparse_matrices = {}

        for feature_name in self.indices.keys():
            all_sparse_matrices[feature_name] = self.get_similarity_matrix_sparse(
                feature_name, top_k
            )

        return all_sparse_matrices

    def get_exact_similarity(
        self, feature_name: str, person_i: int, person_j: int
    ) -> float:
        """Get exact cosine similarity between two specific people for a feature"""
        if feature_name not in self.embeddings:
            return 0.0

        # Get embeddings (already normalized)
        emb_i = self.embeddings[feature_name][person_i]
        emb_j = self.embeddings[feature_name][person_j]

        # Cosine similarity (dot product of normalized vectors)
        similarity = np.dot(emb_i, emb_j)

        return float(similarity)

    def search_similar_people(
        self, person_index: int, feature_weights: Dict[str, float], top_k: int = 20
    ) -> List[Tuple[int, float]]:
        """Find most similar people across all features with weighted combination"""

        # Collect similarities from all features
        combined_scores = {}

        for feature_name, weight in feature_weights.items():
            if feature_name in self.indices and weight > 0:
                # Get top-k similar for this feature
                similarities, indices = self.find_top_k_similar(
                    feature_name, person_index, top_k * 2
                )

                # Add weighted scores
                for sim_score, other_person in zip(similarities, indices):
                    if other_person not in combined_scores:
                        combined_scores[other_person] = 0.0
                    combined_scores[other_person] += weight * sim_score

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_results[:top_k]


class FAISSGraphMatcher:
    """Graph matcher using FAISS for efficient similarity computation"""

    def __init__(self, similarity_engine: FAISSSimilarityEngine):
        self.similarity_engine = similarity_engine

    def create_sparse_similarity_graph(
        self,
        feature_weights: Dict[str, float],
        top_k: int = 50,
        min_similarity: float = 0.1,
    ) -> Dict[Tuple[int, int], float]:
        """Create sparse similarity graph using FAISS indices"""
        logger.info(
            f"🕸️ Creating sparse similarity graph with FAISS (top-{top_k} per person)"
        )

        # Get sparse matrices for all features
        sparse_matrices = self.similarity_engine.build_all_sparse_matrices(top_k)

        # Combine sparse matrices with weights
        combined_edges = {}

        # Get all unique pairs from all feature matrices
        all_pairs = set()
        for feature_matrix in sparse_matrices.values():
            all_pairs.update(feature_matrix.keys())

        logger.info(f"🔢 Processing {len(all_pairs)} unique person pairs...")

        # Calculate weighted similarity for each pair
        for pair in all_pairs:
            person_i, person_j = pair
            combined_similarity = 0.0

            # Sum weighted similarities across features
            for feature_name, weight in feature_weights.items():
                if (
                    feature_name in sparse_matrices
                    and pair in sparse_matrices[feature_name]
                ):
                    feature_sim = sparse_matrices[feature_name][pair]
                    combined_similarity += weight * feature_sim

            # Only keep edges above threshold
            if combined_similarity >= min_similarity:
                combined_edges[pair] = combined_similarity

        logger.info(
            f"✅ Generated {len(combined_edges)} edges above similarity threshold {min_similarity}"
        )

        return combined_edges

    def get_similarity_pairs(
        self, feature_name: str, top_k: int = 50, min_similarity: float = 0.05
    ) -> Dict[Tuple[int, int], float]:
        """Get similarity pairs for a single feature using FAISS"""

        # Get sparse matrix for this specific feature
        sparse_matrices = self.similarity_engine.build_all_sparse_matrices(top_k)

        if feature_name not in sparse_matrices:
            logger.info(f"⚠️ Feature {feature_name} not found in FAISS indices")
            return {}

        feature_matrix = sparse_matrices[feature_name]

        # Filter by minimum similarity
        filtered_pairs = {
            pair: similarity
            for pair, similarity in feature_matrix.items()
            if similarity >= min_similarity
        }

        logger.info(
            f"🔍 Found {len(filtered_pairs)} {feature_name} pairs above {min_similarity} threshold"
        )

        return filtered_pairs


# Example usage and testing
if __name__ == "__main__":
    # Test with dummy data
    logger.info("🧪 Testing FAISS Similarity Engine")

    # Create dummy embeddings (1000 people, 1536 dimensions)
    num_people = 1000
    embedding_dim = 1536

    feature_embeddings = {
        "role_spec": np.random.randn(num_people, embedding_dim).astype(np.float32),
        "experience": np.random.randn(num_people, embedding_dim).astype(np.float32),
        "industry": np.random.randn(num_people, embedding_dim).astype(np.float32),
    }

    # Initialize and build indices
    engine = FAISSSimilarityEngine(embedding_dim)
    engine.build_indices(feature_embeddings)

    # Test similarity search
    person_idx = 0
    similarities, indices = engine.find_top_k_similar("role_spec", person_idx, k=10)
    logger.info(f"\nTop 10 most similar people to person {person_idx} (role_spec):")
    for i, (sim, idx) in enumerate(zip(similarities, indices)):
        logger.info(f"  {i+1}. Person {idx}: similarity = {sim:.3f}")

    # Test graph creation
    feature_weights = {"role_spec": 0.4, "experience": 0.3, "industry": 0.3}
    graph_matcher = FAISSGraphMatcher(engine)
    sparse_graph = graph_matcher.create_sparse_similarity_graph(
        feature_weights, top_k=20
    )

    logger.info(f"\nSparse graph statistics:")
    logger.info(f"  Total edges: {len(sparse_graph)}")
    logger.info(
        f"  Density reduction: {len(sparse_graph)} vs {num_people * (num_people - 1) // 2} possible edges"
    )
    logger.info(
        f"  Compression ratio: {len(sparse_graph) / (num_people * (num_people - 1) // 2) * 100:.2f}%"
    )
