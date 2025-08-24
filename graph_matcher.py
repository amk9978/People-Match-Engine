#!/usr/bin/env python3

import asyncio
import heapq
import os
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from causal_relationship_analyzer import CausalRelationshipAnalyzer
from embedding_service import embedding_service
from faiss_similarity_engine import FAISSGraphMatcher, FAISSSimilarityEngine
from redis_cache import RedisEmbeddingCache
from semantic_person_deduplicator import SemanticPersonDeduplicator
from tag_extractor import tag_extractor

load_dotenv()


class GraphMatcher:
    def __init__(self, csv_path: str, min_density: float = None):
        self.csv_path = csv_path
        self.min_density = min_density or float(os.getenv("min_density", 0.1))
        self.df = None
        self.person_vectors = None
        self.graph = None
        self.cache = RedisEmbeddingCache()
        self.causal_analyzer = CausalRelationshipAnalyzer()
        self.person_deduplicator = SemanticPersonDeduplicator()

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the dataset"""
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} people from dataset")
        return self.df

    async def preprocess_tags(
        self,
        similarity_threshold: float = 0.85,
        fuzzy_threshold: float = 0.90,
        force_rebuild: bool = False,
    ) -> Dict[str, any]:
        """Run tag deduplication preprocessing"""
        print("Running tag deduplication preprocessing...")

        # Check if person-level deduplication results already exist
        existing_stats = self.person_deduplicator.get_stats()
        if not force_rebuild and "error" not in existing_stats:
            print(f"✓ Found existing person-level deduplication results")
            return existing_stats

        # Run semantic person-level deduplication
        print("Building semantic person-level tag deduplication mappings...")
        dedup_results = await self.person_deduplicator.process_dataset_semantic(
            self.csv_path, similarity_threshold
        )

        print(f"✓ Person-level tag deduplication complete")

        return dedup_results

    def extract_tags(self, persona_titles: str) -> List[str]:
        """Extract and clean tags from the persona titles column"""
        return tag_extractor.extract_persona_tags(persona_titles)

    async def extract_and_deduplicate_tags(self, text: str, category: str) -> List[str]:
        """Extract tags and apply semantic deduplication for any feature category"""
        raw_tags = tag_extractor.extract_tags(text, category)

        # Apply semantic person-level deduplication (preserves distinct roles)
        return await self.person_deduplicator.apply_semantic_deduplication(
            raw_tags, category
        )

    async def get_cached_embedding(self, tag: str) -> List[float]:
        """Get embedding using shared embedding service"""
        return await embedding_service.get_embedding(tag)

    async def extract_business_tags_for_person(self, row) -> Dict[str, List[str]]:
        """Extract deduplicated business tags for a person for causal analysis"""
        business_tags = {}

        # Industry (with deduplication)
        industry_text = row["Company Identity - Industry Classification"]
        business_tags["industry"] = await self.extract_and_deduplicate_tags(
            industry_text, "industry"
        )

        # Market (with deduplication)
        market_text = row["Company Market - Market Traction"]
        business_tags["market"] = await self.extract_and_deduplicate_tags(
            market_text, "market"
        )

        # Offering (with deduplication)
        offering_text = row["Company Offering - Value Proposition"]
        business_tags["offering"] = await self.extract_and_deduplicate_tags(
            offering_text, "offering"
        )

        return business_tags

    async def embed_features(self) -> Dict[str, np.ndarray]:
        """Create OpenAI embeddings for multiple feature categories with async batch processing"""
        print("Creating feature embeddings...")

        feature_columns = {
            "role_spec": "Professional Identity - Role Specification",
            "experience": "Professional Identity - Experience Level",
            "industry": "Company Identity - Industry Classification",
            "market": "Company Market - Market Traction",
            "offering": "Company Offering - Value Proposition",
            "personas": "All Persona Titles",
        }

        feature_embeddings = {}

        for feature_name, column_name in feature_columns.items():
            print(f"\nProcessing {feature_name} ({column_name})...")

            # Step 1: Collect all unique values for this feature (with deduplication for ALL features)
            all_unique_values = set()
            for _, row in self.df.iterrows():
                # Apply deduplication to ALL feature categories
                values = await self.extract_and_deduplicate_tags(
                    row[column_name], feature_name
                )
                all_unique_values.update(values)

            print(f"Found {len(all_unique_values)} unique values for {feature_name}")

            # Step 2: Separate cached and uncached values
            cached_values = {}
            uncached_values = []
            cache_hits = 0

            for value in all_unique_values:
                if value.strip():
                    if self.cache.exists(value):
                        cached_values[value] = self.cache.get(value)
                        cache_hits += 1
                    else:
                        uncached_values.append(value)
                else:
                    cached_values[value] = [0.0] * 1536

            print(
                f"  Cache hits: {cache_hits}, API calls needed: {len(uncached_values)}"
            )

            # Step 3: Batch process uncached values asynchronously
            value_embeddings = cached_values.copy()

            if uncached_values:
                print(
                    f"  Processing {len(uncached_values)} embeddings in async batches..."
                )

                # Create async tasks for all uncached values
                tasks = []
                for value in uncached_values:
                    tasks.append(self.get_cached_embedding(value))

                # Execute all tasks concurrently
                embeddings_results = await asyncio.gather(
                    *tasks, return_exceptions=True
                )

                # Map results back to values
                for value, embedding in zip(uncached_values, embeddings_results):
                    if isinstance(embedding, Exception):
                        print(f"  Error processing {value}: {embedding}")
                        value_embeddings[value] = [0.0] * 1536
                    else:
                        value_embeddings[value] = embedding

            print(
                f"  {feature_name} summary: {cache_hits} from cache, {len(uncached_values)} async API calls"
            )

            # Step 3: Create person vectors for this feature (using deduplicated tags for ALL features)
            person_feature_embeddings = []
            for idx, row in self.df.iterrows():
                # Apply deduplication to ALL feature categories
                values = await self.extract_and_deduplicate_tags(
                    row[column_name], feature_name
                )

                if values:
                    # Get embeddings for all values of this person's feature
                    person_value_embeddings = [
                        value_embeddings[val]
                        for val in values
                        if val in value_embeddings
                    ]

                    if person_value_embeddings:
                        # L2 normalized sum
                        person_value_embeddings = np.array(person_value_embeddings)
                        person_embedding = np.sum(person_value_embeddings, axis=0)
                        norm = np.linalg.norm(person_embedding)
                        if norm > 0:
                            person_embedding = person_embedding / norm
                    else:
                        person_embedding = [0.0] * 1536
                else:
                    person_embedding = [0.0] * 1536

                person_feature_embeddings.append(person_embedding)

            feature_embeddings[feature_name] = np.array(person_feature_embeddings)
            print(
                f"Created {feature_embeddings[feature_name].shape[1]}D embeddings for {len(person_feature_embeddings)} people"
            )

        # Print cache info
        cache_info = self.cache.get_cache_info()
        print(f"Redis cache status: {cache_info}")

        return feature_embeddings

    def create_graph(self, feature_embeddings: Dict[str, np.ndarray]) -> nx.Graph:
        """Create a weighted graph with hybrid professional similarity + business complementarity"""
        print(
            "Creating hybrid similarity graph (professional similarity + business complementarity)..."
        )

        self.graph = nx.Graph()

        # Add nodes (people)
        for idx, row in self.df.iterrows():
            self.graph.add_node(
                idx, name=row["Person Name"], company=row["Person Company"]
            )

        # Load causal relationship graph
        if not self.causal_analyzer.load_causal_graph_from_redis():
            print(
                "Warning: No causal relationship graph found in Redis. Building first..."
            )
            print("Run: python causal_relationship_analyzer.py")
            print("Falling back to traditional similarity calculation...")
            return self.create_graph_traditional(feature_embeddings)

        # Load persona complementarity matrix
        if not self.causal_analyzer.load_persona_matrix_from_redis():
            print(
                "Warning: No persona complementarity matrix found. Persona complementarity will use fallback."
            )
            # Continue anyway - we can still use other components

        num_people = len(self.df)

        # Calculate PROFESSIONAL similarities (homophily - similar people connect)
        professional_features = ["role_spec", "experience", "personas"]
        professional_similarities = {}

        for feature_name in professional_features:
            if feature_name in feature_embeddings:
                similarity_matrix = cosine_similarity(feature_embeddings[feature_name])
                professional_similarities[feature_name] = similarity_matrix
                print(f"Calculated {feature_name} professional similarities")

        # Professional similarity weights (should sum to 1.0)
        professional_weights = {
            "role_spec": 0.5,  # Role is most important for professional chemistry
            "experience": 0.35,  # Experience level compatibility
            "personas": 0.15,  # Personas add context
        }

        # Calculate hybrid similarities
        for i in range(num_people):
            for j in range(i + 1, num_people):

                # 1. PROFESSIONAL SIMILARITY (homophily)
                professional_score = 0.0
                for feature_name, weight in professional_weights.items():
                    if feature_name in professional_similarities:
                        feature_sim = professional_similarities[feature_name][i][j]
                        professional_score += weight * feature_sim

                # 2. BUSINESS COMPLEMENTARITY (from causal relationships)
                person1_business_tags = self.extract_business_tags_for_person(
                    self.df.iloc[i]
                )
                person2_business_tags = self.extract_business_tags_for_person(
                    self.df.iloc[j]
                )

                business_complementarity = (
                    self.causal_analyzer.calculate_business_complementarity_fast(
                        person1_business_tags, person2_business_tags
                    )
                )

                # 3. EXPERIENCE SIMILARITY (peer-level networking)
                experience_similarity = 0.0
                if "experience" in feature_embeddings:
                    experience_similarity = cosine_similarity(
                        feature_embeddings["experience"]
                    )[i][j]

                # 4. EXPERIENCE COMPLEMENTARITY (senior-junior mentorship)
                experience_complementarity = 1.0 - experience_similarity

                # 4. PERSONA COMPLEMENTARITY (ChatGPT-based strategic complementarity)
                person1_personas = self.extract_tags(
                    self.df.iloc[i].get("All Persona Titles", "")
                )
                person2_personas = self.extract_tags(
                    self.df.iloc[j].get("All Persona Titles", "")
                )
                persona_complementarity = (
                    self.causal_analyzer.calculate_persona_complementarity_fast(
                        person1_personas, person2_personas
                    )
                )

                # 5. EXPANDED HYBRID SCORE (5 components with both experience similarity and complementarity)
                role_weight = float(os.getenv("ROLE_SIMILARITY_WEIGHT", "0.30"))
                exp_sim_weight = float(
                    os.getenv("EXPERIENCE_SIMILARITY_WEIGHT", "0.15")
                )
                exp_comp_weight = float(
                    os.getenv("EXPERIENCE_COMPLEMENTARITY_WEIGHT", "0.15")
                )
                business_comp_weight = float(
                    os.getenv("BUSINESS_COMPLEMENTARITY_WEIGHT", "0.25")
                )
                persona_comp_weight = float(
                    os.getenv("PERSONA_COMPLEMENTARITY_WEIGHT", "0.15")
                )

                # Role similarity from professional_score (weighted by role_spec component)
                role_similarity = professional_score * (
                    professional_weights.get("role_spec", 0.5)
                    / sum(professional_weights.values())
                )

                hybrid_similarity = (
                    role_weight * role_similarity
                    + exp_sim_weight * experience_similarity
                    + exp_comp_weight * experience_complementarity
                    + business_comp_weight * business_complementarity
                    + persona_comp_weight * persona_complementarity
                )

                # Only add edges above threshold
                if (
                    hybrid_similarity > 0.1
                ):  # Lowered threshold since we now have complementarity
                    self.graph.add_edge(
                        i,
                        j,
                        weight=hybrid_similarity,
                        role_similarity=role_similarity,
                        experience_similarity=experience_similarity,
                        experience_complementarity=experience_complementarity,
                        business_complementarity=business_complementarity,
                        persona_complementarity=persona_complementarity,
                    )

        print(
            f"Created hybrid graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
        )
        print(
            "Hybrid weights: 70% professional similarity + 30% business complementarity"
        )
        print(f"Professional weights: {professional_weights}")
        return self.graph

    def create_graph_traditional(
        self, feature_embeddings: Dict[str, np.ndarray]
    ) -> nx.Graph:
        """Fallback: Create traditional similarity-only graph"""
        print("Creating traditional similarity-only graph...")

        # Calculate pairwise similarities for each feature
        feature_similarities = {}
        for feature_name, embeddings in feature_embeddings.items():
            similarity_matrix = cosine_similarity(embeddings)
            feature_similarities[feature_name] = similarity_matrix
            print(f"Calculated {feature_name} similarities")

        num_people = len(self.df)

        # Apply 5-component similarity equation (consistent with main create_graph)
        for i in range(num_people):
            for j in range(i + 1, num_people):

                # 1. ROLE SIMILARITY (professional homophily)
                role_similarity = 0.0
                if "role_spec" in feature_similarities:
                    role_similarity = feature_similarities["role_spec"][i][j]

                # 2. EXPERIENCE SIMILARITY (peer-level networking)
                experience_similarity = 0.0
                if "experience" in feature_similarities:
                    experience_similarity = feature_similarities["experience"][i][j]

                # 3. EXPERIENCE COMPLEMENTARITY (senior-junior mentorship)
                experience_complementarity = 1.0 - experience_similarity

                # 3. BUSINESS COMPLEMENTARITY (market/offering synergies - simplified without causal graph)
                business_complementarity = 0.0
                if (
                    "market" in feature_similarities
                    and "offering" in feature_similarities
                ):
                    market_comp = 1.0 - feature_similarities["market"][i][j]
                    offering_comp = 1.0 - feature_similarities["offering"][i][j]
                    business_complementarity = (market_comp + offering_comp) / 2

                # 4. PERSONA COMPLEMENTARITY (ChatGPT-based strategic complementarity)
                person1_personas = self.extract_tags(
                    self.df.iloc[i].get("All Persona Titles", "")
                )
                person2_personas = self.extract_tags(
                    self.df.iloc[j].get("All Persona Titles", "")
                )
                persona_complementarity = (
                    self.causal_analyzer.calculate_persona_complementarity_fast(
                        person1_personas, person2_personas
                    )
                )

                # Apply 5-component equation with both experience similarity and complementarity
                role_weight = float(os.getenv("ROLE_SIMILARITY_WEIGHT", "0.30"))
                exp_sim_weight = float(
                    os.getenv("EXPERIENCE_SIMILARITY_WEIGHT", "0.15")
                )
                exp_comp_weight = float(
                    os.getenv("EXPERIENCE_COMPLEMENTARITY_WEIGHT", "0.15")
                )
                business_comp_weight = float(
                    os.getenv("BUSINESS_COMPLEMENTARITY_WEIGHT", "0.25")
                )
                persona_comp_weight = float(
                    os.getenv("PERSONA_COMPLEMENTARITY_WEIGHT", "0.15")
                )

                hybrid_similarity = (
                    role_weight * role_similarity
                    + exp_sim_weight * experience_similarity
                    + exp_comp_weight * experience_complementarity
                    + business_comp_weight * business_complementarity
                    + persona_comp_weight * persona_complementarity
                )

                if hybrid_similarity > 0.1:
                    self.graph.add_edge(
                        i,
                        j,
                        weight=hybrid_similarity,
                        role_similarity=role_similarity,
                        experience_similarity=experience_similarity,
                        experience_complementarity=experience_complementarity,
                        business_complementarity=business_complementarity,
                        persona_complementarity=persona_complementarity,
                    )

        print(
            f"Created traditional graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
        )
        return self.graph

    def calculate_subgraph_density(self, nodes: Set[int]) -> float:
        """Calculate the weighted density of a subgraph"""
        if len(nodes) < 2:
            return 0.0

        subgraph = self.graph.subgraph(nodes)
        total_weight = sum(data["weight"] for _, _, data in subgraph.edges(data=True))
        max_possible_edges = len(nodes) * (len(nodes) - 1) / 2

        if max_possible_edges == 0:
            return 0.0

        return total_weight / max_possible_edges

    def densest_subgraph_peeling(self, find_all: bool = False) -> List[Set[int]]:
        """
        Implement weighted densest subgraph peeling algorithm using min-heap
        Returns subgraphs in order of decreasing density
        """
        print("Running densest subgraph peeling...")

        remaining_nodes = set(self.graph.nodes())
        subgraphs = []

        # Pre-calculate initial weighted degrees for all nodes
        node_degrees = {}
        for node in remaining_nodes:
            node_degrees[node] = sum(
                self.graph[node][neighbor]["weight"]
                for neighbor in self.graph.neighbors(node)
                if neighbor in remaining_nodes
            )

        # Create min-heap with (degree, node) pairs
        heap = [(degree, node) for node, degree in node_degrees.items()]
        heapq.heapify(heap)

        while len(remaining_nodes) > 1:
            # Calculate and store current density
            current_density = self.calculate_subgraph_density(remaining_nodes)

            # If density is above threshold, save this subgraph
            if current_density >= self.min_density:
                subgraphs.append((remaining_nodes.copy(), current_density))
                print(
                    f"Found subgraph with {len(remaining_nodes)} nodes and density {current_density:.4f}"
                )
                if not find_all:
                    break  # Stop at first (largest) dense subgraph if find_all=False

            # Find the node with minimum degree from heap
            while heap:
                min_degree, node_to_remove = heapq.heappop(heap)

                # Check if this node is still in the graph and degree is current
                if (
                    node_to_remove in remaining_nodes
                    and node_degrees[node_to_remove] == min_degree
                ):
                    break
            else:
                # No valid nodes left
                break

            # Update degrees of neighbors before removing the node
            neighbors_to_update = []
            for neighbor in self.graph.neighbors(node_to_remove):
                if neighbor in remaining_nodes and neighbor != node_to_remove:
                    edge_weight = self.graph[node_to_remove][neighbor]["weight"]
                    node_degrees[neighbor] -= edge_weight
                    neighbors_to_update.append(neighbor)

            # Remove the node
            remaining_nodes.remove(node_to_remove)
            del node_degrees[node_to_remove]

            # Add updated neighbors back to heap
            for neighbor in neighbors_to_update:
                heapq.heappush(heap, (node_degrees[neighbor], neighbor))

        # Sort by density (descending) and return just the node sets
        subgraphs.sort(key=lambda x: x[1], reverse=True)
        return [nodes for nodes, density in subgraphs]

    def find_largest_dense_subgraph(self) -> Tuple[Set[int], float]:
        """Find the largest subgraph with density above min_density"""
        dense_subgraphs = self.densest_subgraph_peeling()

        if not dense_subgraphs:
            print(f"No subgraphs found with density >= {self.min_density}")
            # Return all nodes as fallback
            all_nodes = set(self.graph.nodes())
            fallback_density = self.calculate_subgraph_density(all_nodes)
            print(
                f"Fallback: returning all {len(all_nodes)} nodes with density {fallback_density:.4f}"
            )
            return all_nodes, fallback_density

        # Find the largest among dense subgraphs
        largest_subgraph = max(dense_subgraphs, key=len)
        density = self.calculate_subgraph_density(largest_subgraph)

        print(
            f"Largest dense subgraph: {len(largest_subgraph)} nodes with density {density:.4f}"
        )

        return largest_subgraph, density

    def calculate_centroid_and_insights(
        self, nodes: Set[int], feature_embeddings: Dict[str, np.ndarray]
    ) -> Dict:
        """Calculate subgraph centroid and find most representative tags"""
        if not nodes:
            return {}

        insights = {}

        # Calculate centroid for each feature
        for feature_name, embeddings in feature_embeddings.items():
            subgraph_embeddings = embeddings[list(nodes)]
            centroid = np.mean(subgraph_embeddings, axis=0)

            # Find closest tags to centroid
            closest_tags = self.find_closest_tags_to_centroid(centroid, feature_name)

            insights[feature_name] = {
                "centroid": centroid,
                "closest_tags": closest_tags[:5],  # Top 5 closest tags
                "centroid_norm": np.linalg.norm(centroid),
            }

        return insights

    def find_closest_tags_to_centroid(
        self, centroid: np.ndarray, feature_name: str
    ) -> List[Tuple[str, float]]:
        """Find tags closest to the centroid for a specific feature"""
        tag_similarities = []

        # Get all cached embeddings for this feature type
        # We'll iterate through our data to find relevant tags
        feature_columns = {
            "role_spec": "Professional Identity - Role Specification",
            "experience": "Professional Identity - Experience Level",
            "industry": "Company Identity - Industry Classification",
            "market": "Company Market - Market Traction",
            "offering": "Company Offering - Value Proposition",
            "personas": "All Persona Titles",
        }

        if feature_name not in feature_columns:
            return []

        column_name = feature_columns[feature_name]
        seen_tags = set()

        # Collect all unique tags for this feature from the dataset
        for _, row in self.df.iterrows():
            if feature_name == "personas":
                values = self.extract_tags(row[column_name])
            else:
                text = str(row[column_name]) if pd.notna(row[column_name]) else ""
                values = [val.strip() for val in text.split("|") if val.strip()]

            for tag in values:
                if tag not in seen_tags:
                    seen_tags.add(tag)
                    # Get cached embedding
                    tag_embedding = self.cache.get(tag)
                    if tag_embedding is not None:
                        similarity = np.dot(centroid, tag_embedding) / (
                            np.linalg.norm(centroid) * np.linalg.norm(tag_embedding)
                            + 1e-8
                        )
                        tag_similarities.append((tag, similarity))

        # Sort by similarity (highest first)
        tag_similarities.sort(key=lambda x: x[1], reverse=True)
        return tag_similarities

    def get_subgraph_info(
        self, nodes: Set[int], feature_embeddings: Dict[str, np.ndarray] = None
    ) -> Dict:
        """Get detailed information about a subgraph including centroid insights"""
        if not nodes:
            return {}

        people_info = []
        for node in nodes:
            row = self.df.iloc[node]
            people_info.append(
                {
                    "name": row["Person Name"],
                    "title": row["Person Title"],
                    "company": row["Person Company"],
                    "tags": self.extract_tags(row["All Persona Titles"]),
                }
            )

        result = {
            "size": len(nodes),
            "density": self.calculate_subgraph_density(nodes),
            "people": people_info,
        }

        if feature_embeddings:
            centroid_insights = self.calculate_centroid_and_insights(
                nodes, feature_embeddings
            )
            result["centroid_insights"] = centroid_insights

        return result

    async def run_analysis(self) -> Dict:
        """Run complete analysis pipeline with tag deduplication"""
        print(
            "Starting multi-feature graph matching analysis with tag deduplication..."
        )

        # Step 1: Load data
        self.load_data()

        # Step 2: Preprocess tags (deduplication)
        dedup_stats = await self.preprocess_tags()

        # Step 3: Create embeddings (now using deduplicated tags)
        feature_embeddings = await self.embed_features()

        # Step 4: Create hybrid similarity graph
        self.create_graph(feature_embeddings)

        largest_dense_nodes, density = self.find_largest_dense_subgraph()

        result = self.get_subgraph_info(largest_dense_nodes, feature_embeddings)

        print(f"\nAnalysis complete!")
        print(f"Found largest dense subgraph with {result['size']} people")
        print(f"Density: {result['density']:.4f} (threshold: {self.min_density})")

        if "centroid_insights" in result:
            print(f"\n🎯 CENTROID INSIGHTS - Most Representative Tags:")
            print("=" * 60)

            for feature_name, insights in result["centroid_insights"].items():
                print(f"\n📊 {feature_name.upper()}:")
                for tag, similarity in insights["closest_tags"]:
                    print(f"  • {tag:<50} (similarity: {similarity:.3f})")

        recommendations = self.get_expansion_recommendations(
            largest_dense_nodes, feature_embeddings
        )
        if recommendations:
            print(f"\n💡 EXPANSION RECOMMENDATIONS:")
            print("=" * 60)
            print("People who might improve subgraph density:")
            for i, rec in enumerate(recommendations, 1):
                print(
                    f"{i}. {rec['name']} ({rec['title']} at {rec['company']}) - Score: {rec['similarity_score']:.3f}"
                )

        return result

    def get_expansion_recommendations(
        self, nodes: Set[int], feature_embeddings: Dict[str, np.ndarray], top_n: int = 3
    ) -> List[Dict]:
        """Find people not in subgraph who might improve density"""
        if not nodes or len(nodes) == len(self.df):
            return []

        centroids = {}
        for feature_name, embeddings in feature_embeddings.items():
            subgraph_embeddings = embeddings[list(nodes)]
            centroids[feature_name] = np.mean(subgraph_embeddings, axis=0)

        candidates = []
        feature_weights = {
            "role_spec": 0.25,
            "experience": 0.15,
            "industry": 0.20,
            "market": 0.15,
            "offering": 0.15,
            "personas": 0.10,
        }

        for idx in range(len(self.df)):
            if idx not in nodes:
                total_similarity = 0.0
                for feature_name, weight in feature_weights.items():
                    if feature_name in centroids and feature_name in feature_embeddings:
                        person_embedding = feature_embeddings[feature_name][idx]
                        centroid = centroids[feature_name]

                        similarity = np.dot(centroid, person_embedding) / (
                            np.linalg.norm(centroid) * np.linalg.norm(person_embedding)
                            + 1e-8
                        )
                        total_similarity += weight * similarity

                row = self.df.iloc[idx]
                candidates.append(
                    {
                        "index": idx,
                        "name": row["Person Name"],
                        "title": row["Person Title"],
                        "company": row["Person Company"],
                        "similarity_score": total_similarity,
                    }
                )

        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:top_n]

    def create_graph_faiss(
        self, feature_embeddings: Dict[str, np.ndarray], top_k: int = 50
    ) -> nx.Graph:
        """Create graph using FAISS for efficient similarity computation"""
        print(f"🚀 Creating FAISS-optimized graph (top-{top_k} per person)...")

        self.graph = nx.Graph()

        # Add nodes (people)
        for idx, row in self.df.iterrows():
            self.graph.add_node(
                idx, name=row["Person Name"], company=row["Person Company"]
            )

        # Initialize FAISS similarity engine
        faiss_engine = FAISSSimilarityEngine(embedding_dim=1536)
        faiss_engine.build_indices(feature_embeddings)

        # Get hyperparameters for 5-component equation
        role_weight = float(os.getenv("ROLE_SIMILARITY_WEIGHT", "0.30"))
        exp_sim_weight = float(os.getenv("EXPERIENCE_SIMILARITY_WEIGHT", "0.15"))
        exp_comp_weight = float(os.getenv("EXPERIENCE_COMPLEMENTARITY_WEIGHT", "0.15"))
        business_comp_weight = float(
            os.getenv("BUSINESS_COMPLEMENTARITY_WEIGHT", "0.25")
        )
        persona_comp_weight = float(os.getenv("PERSONA_COMPLEMENTARITY_WEIGHT", "0.15"))

        # Create sparse similarity matrices using FAISS
        feature_weights_similarity = {
            "role_spec": role_weight,
            "experience": exp_sim_weight,
            "personas": persona_comp_weight,  # For similarity component
        }

        # Use FAISS to get sparse similarities for similarity-based features
        faiss_matcher = FAISSGraphMatcher(faiss_engine)
        sparse_similarities = faiss_matcher.create_sparse_similarity_graph(
            feature_weights_similarity, top_k=top_k, min_similarity=0.05
        )

        # Load causal/complementarity analyzers
        if not self.causal_analyzer.load_causal_graph_from_redis():
            print(
                "Warning: No causal relationship graph found. Using FAISS similarities only."
            )
            # Create edges from FAISS similarities
            for (person_i, person_j), similarity in sparse_similarities.items():
                if similarity > 0.1:
                    self.graph.add_edge(person_i, person_j, weight=similarity)

            print(
                f"Created FAISS-only graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
            )
            return self.graph

        # Load persona complementarity matrix
        if not self.causal_analyzer.load_persona_matrix_from_redis():
            print("Warning: No persona complementarity matrix found.")

        print("🔧 Adding complementarity components to FAISS similarities...")

        # For each sparse similarity pair, add complementarity components
        enhanced_edges = {}

        for (person_i, person_j), base_similarity in sparse_similarities.items():
            # Get complementarity components

            # Experience complementarity
            if "experience" in feature_embeddings:
                exp_similarity = faiss_engine.get_exact_similarity(
                    "experience", person_i, person_j
                )
                experience_complementarity = 1.0 - exp_similarity
            else:
                experience_complementarity = 0.0

            # Business complementarity
            person1_business_tags = self.extract_business_tags_for_person(
                self.df.iloc[person_i]
            )
            person2_business_tags = self.extract_business_tags_for_person(
                self.df.iloc[person_j]
            )
            business_complementarity = (
                self.causal_analyzer.calculate_business_complementarity_fast(
                    person1_business_tags, person2_business_tags
                )
            )

            # Persona complementarity
            person1_personas = self.extract_tags(
                self.df.iloc[person_i].get("All Persona Titles", "")
            )
            person2_personas = self.extract_tags(
                self.df.iloc[person_j].get("All Persona Titles", "")
            )
            persona_complementarity = (
                self.causal_analyzer.calculate_persona_complementarity_fast(
                    person1_personas, person2_personas
                )
            )

            # Apply 5-component hybrid equation
            hybrid_similarity = (
                base_similarity  # Already includes role + experience similarity + personas
                + exp_comp_weight * experience_complementarity
                + business_comp_weight * business_complementarity
                + persona_comp_weight
                * persona_complementarity  # Add complementarity component
            )

            if hybrid_similarity > 0.1:
                enhanced_edges[(person_i, person_j)] = hybrid_similarity

        # Create final graph edges
        for (person_i, person_j), weight in enhanced_edges.items():
            self.graph.add_edge(person_i, person_j, weight=weight)

        print(
            f"✅ Created FAISS-optimized graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
        )
        print(
            f"🎯 Efficiency: {len(enhanced_edges)} edges vs {len(self.df) * (len(self.df) - 1) // 2} possible ({len(enhanced_edges) / (len(self.df) * (len(self.df) - 1) // 2) * 100:.1f}% density)"
        )

        return self.graph


async def main():
    # Initialize matcher
    matcher = GraphMatcher(
        "/home/ryan/PycharmProjects/match_engine/data/test_batch2.csv"
    )

    # Run analysis
    result = await matcher.run_analysis()

    # Print results
    print("\n" + "=" * 50)
    print("DENSEST SUBGRAPH RESULTS")
    print("=" * 50)

    if result["people"]:
        for i, person in enumerate(result["people"], 1):
            print(f"\n{i}. {person['name']}")
            print(f"   Title: {person['title']}")
            print(f"   Company: {person['company']}")
            print(f"   Tags: {', '.join(person['tags'][:3])}...")  # Show first 3 tags
    else:
        print("No dense subgraph found above the minimum density threshold.")


if __name__ == "__main__":
    asyncio.run(main())
