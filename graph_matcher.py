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
        """Load and preprocess the dataset with row filtering"""
        self.df = pd.read_csv(self.csv_path)
        original_count = len(self.df)
        print(f"Loaded {original_count} people from dataset")
        
        # Filter out rows with empty essential columns
        self.df = self.filter_incomplete_rows(self.df)
        filtered_count = len(self.df)
        
        if filtered_count < original_count:
            removed_count = original_count - filtered_count
            print(f"🗑️ Filtered out {removed_count} rows with missing essential data")
            print(f"📊 Using {filtered_count} complete rows for analysis")
        
        return self.df
    
    def filter_incomplete_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out rows with empty essential columns"""
        
        # Define essential columns that must be non-empty
        essential_columns = [
            "Person Name",
            "Person Title", 
            "Person Company",
            "Person Linkedin URL",  # LinkedIn URL is essential
            "Professional Identity - Role Specification",
            "Professional Identity - Experience Level",
            "Company Identity - Industry Classification",
            "Company Market - Market Traction",
            "Company Offering - Value Proposition",
            "All Persona Titles"
        ]
        
        # Check which essential columns exist in the dataset
        existing_columns = [col for col in essential_columns if col in df.columns]
        missing_columns = [col for col in essential_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ Warning: Missing columns in dataset: {missing_columns}")
        
        # Filter rows where any essential column is empty/NaN
        initial_count = len(df)
        
        # Create a mask for rows that have non-empty values in all essential columns
        mask = pd.Series([True] * len(df), index=df.index)
        
        for column in existing_columns:
            # Check for NaN, empty strings, or whitespace-only strings
            column_mask = (
                df[column].notna() & 
                (df[column].astype(str).str.strip() != '') &
                (df[column].astype(str).str.strip() != 'nan')
            )
            mask = mask & column_mask
            
            # Show which rows would be filtered by this column
            filtered_by_column = (~column_mask).sum()
            if filtered_by_column > 0:
                print(f"   • {column}: {filtered_by_column} rows have empty/missing values")
        
        # Apply the filter
        filtered_df = df[mask].copy()
        
        return filtered_df

    async def preprocess_tags(
        self,
        similarity_threshold: float = 0.7,
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
        return raw_tags

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

    async def create_graph(self, feature_embeddings: Dict[str, np.ndarray]) -> nx.Graph:
        """Create a weighted graph using optimized 5-component hybrid equation"""
        
        # Check if FAISS optimization is enabled
        use_faiss = os.getenv("USE_FAISS_OPTIMIZATION", "false").lower() == "true"
        
        if use_faiss:
            print("🚀 Using FAISS-optimized graph creation...")
            return await self.create_graph_faiss(feature_embeddings)
        else:
            print("⚡ Using optimized 5-component hybrid similarity graph...")
            return await self.create_graph_optimized(feature_embeddings)

    async def create_graph_optimized(self, feature_embeddings: Dict[str, np.ndarray]) -> nx.Graph:
        """Optimized graph creation with precomputed similarities and tags"""
        
        self.graph = nx.Graph()
        num_people = len(self.df)

        # Add nodes (people)
        for idx, row in self.df.iterrows():
            self.graph.add_node(
                idx, name=row["Person Name"], company=row["Person Company"]
            )

        # Load/build all complementarity matrices
        if not self.causal_analyzer.load_causal_graph_from_redis():
            print("🔧 Business complementarity matrix not found. Building automatically...")
            try:
                await self.causal_analyzer.build_causal_relationship_graph(self.csv_path)
                print("✅ Business complementarity matrix built and cached")
            except Exception as e:
                print(f"⚠️ Error building business matrix: {e}")
                print("Falling back to traditional similarity calculation...")
                return await self.create_graph_traditional(feature_embeddings)

        if not self.causal_analyzer.load_experience_matrix_from_redis():
            print("🔧 Experience complementarity matrix not found. Building automatically...")
            try:
                await self.causal_analyzer.build_experience_complementarity_matrix(self.csv_path)
                print("✅ Experience complementarity matrix built and cached")
            except Exception as e:
                print(f"⚠️ Using embedding-based experience complementarity fallback")

        # Get hyperparameters from environment
        role_weight = float(os.getenv("ROLE_SIMILARITY_WEIGHT", "0.30"))
        exp_sim_weight = float(os.getenv("EXPERIENCE_SIMILARITY_WEIGHT", "0.15"))
        exp_comp_weight = float(os.getenv("EXPERIENCE_COMPLEMENTARITY_WEIGHT", "0.15"))
        business_comp_weight = float(os.getenv("BUSINESS_COMPLEMENTARITY_WEIGHT", "0.25"))
        persona_comp_weight = float(os.getenv("PERSONA_COMPLEMENTARITY_WEIGHT", "0.15"))

        print(f"📊 Using 5-component equation weights:")
        print(f"   Role Similarity: {role_weight:+.2f}")
        print(f"   Experience Similarity: {exp_sim_weight:+.2f}")
        print(f"   Experience Complementarity: {exp_comp_weight:+.2f}")
        print(f"   Business Complementarity: {business_comp_weight:+.2f}")
        print(f"   Persona Complementarity: {persona_comp_weight:+.2f}")
        print(f"   Total: {role_weight + exp_sim_weight + exp_comp_weight + business_comp_weight + persona_comp_weight:.2f}")

        # OPTIMIZATION 1: Precompute similarity matrices once
        print("⚡ Precomputing similarity matrices...")
        role_sim_matrix = None
        exp_sim_matrix = None
        
        if "role_spec" in feature_embeddings:
            role_sim_matrix = cosine_similarity(feature_embeddings["role_spec"])
        
        if "experience" in feature_embeddings:
            exp_sim_matrix = cosine_similarity(feature_embeddings["experience"])

        # OPTIMIZATION 2: Precompute all person tags once
        print("⚡ Precomputing person tags...")
        person_tags_cache = {}
        for idx, row in self.df.iterrows():
            person_tags_cache[idx] = {
                'experience': self.extract_tags(row.get("Professional Identity - Experience Level", "")),
                'personas': self.extract_tags(row.get("All Persona Titles", "")),
                'business': None  # Will compute async below
            }
        
        # Precompute business tags (async)
        print("⚡ Precomputing business tags...")
        for idx, row in self.df.iterrows():
            person_tags_cache[idx]['business'] = await self.extract_business_tags_for_person(row)

        # OPTIMIZATION 3: Vectorized edge calculation with early termination
        print(f"⚡ Computing {num_people * (num_people - 1) // 2} pairwise similarities...")
        edges_added = 0
        
        for i in range(num_people):
            for j in range(i + 1, num_people):
                
                # Component 1: Role Similarity (precomputed matrix)
                role_similarity = role_sim_matrix[i][j] if role_sim_matrix is not None else 0.0
                
                # Component 2: Experience Similarity (precomputed matrix)  
                experience_similarity = exp_sim_matrix[i][j] if exp_sim_matrix is not None else 0.0
                
                # Component 3: Experience Complementarity (cached tags)
                experience_complementarity = await self.causal_analyzer.calculate_experience_complementarity_fast(
                    person_tags_cache[i]['experience'], person_tags_cache[j]['experience']
                )
                
                # Component 4: Business Complementarity (cached tags)
                business_complementarity = self.causal_analyzer.calculate_business_complementarity_fast(
                    person_tags_cache[i]['business'], person_tags_cache[j]['business']
                )
                
                # Component 5: Persona Complementarity (cached tags)
                persona_complementarity = await self.causal_analyzer.calculate_persona_complementarity_fast(
                    person_tags_cache[i]['personas'], person_tags_cache[j]['personas']
                )

                # SINGLE 5-COMPONENT HYBRID EQUATION
                hybrid_similarity = (
                    role_weight * role_similarity +
                    exp_sim_weight * experience_similarity +
                    exp_comp_weight * experience_complementarity +
                    business_comp_weight * business_complementarity +
                    persona_comp_weight * persona_complementarity
                )

                # Only add edges above threshold
                if hybrid_similarity > 0.1:
                    self.graph.add_edge(
                        i, j,
                        weight=hybrid_similarity,
                        role_similarity=role_similarity,
                        experience_similarity=experience_similarity,
                        experience_complementarity=experience_complementarity,
                        business_complementarity=business_complementarity,
                        persona_complementarity=persona_complementarity,
                    )
                    edges_added += 1

        print(f"✅ Created optimized 5-component hybrid graph:")
        print(f"   Nodes: {self.graph.number_of_nodes()}")
        print(f"   Edges: {edges_added}")
        print(f"   Density: {edges_added / (num_people * (num_people - 1) // 2) * 100:.1f}%")
        return self.graph

    async def create_graph_traditional(
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
                    await self.causal_analyzer.calculate_persona_complementarity_fast(
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

    def find_subgroups_in_subgraph(self, nodes: Set[int], min_subgroup_size: int = 3) -> List[Dict]:
        """Find cohesive subgroups within the dense subgraph using edge weights"""
        if len(nodes) < min_subgroup_size * 2:
            return []
            
        subgraph = self.graph.subgraph(nodes)
        
        # Use Louvain community detection for clustering
        try:
            communities = nx.community.louvain_communities(subgraph, weight='weight', resolution=1.2)
        except:
            # Fallback to greedy modularity if Louvain fails
            communities = nx.community.greedy_modularity_communities(subgraph, weight='weight')
        
        subgroups = []
        
        for i, community in enumerate(communities):
            if len(community) >= min_subgroup_size:
                # Calculate internal density and average weight
                community_subgraph = subgraph.subgraph(community)
                
                total_weight = 0
                edge_count = 0
                max_weight = 0
                min_weight = float('inf')
                
                for u, v, data in community_subgraph.edges(data=True):
                    weight = data.get('weight', 0)
                    total_weight += weight
                    edge_count += 1
                    max_weight = max(max_weight, weight)
                    min_weight = min(min_weight, weight)
                
                avg_internal_weight = total_weight / edge_count if edge_count > 0 else 0
                
                # Calculate density specifically for this subgroup
                possible_edges = len(community) * (len(community) - 1) / 2
                internal_density = edge_count / possible_edges if possible_edges > 0 else 0
                
                # Get representative people from this subgroup
                people_in_subgroup = []
                for node in list(community)[:5]:  # Show up to 5 people
                    row = self.df.iloc[node]
                    people_in_subgroup.append({
                        "name": row["Person Name"],
                        "title": row["Person Title"], 
                        "company": row["Person Company"]
                    })
                
                subgroups.append({
                    "subgroup_id": i + 1,
                    "size": len(community),
                    "internal_density": internal_density,
                    "avg_connection_strength": avg_internal_weight,
                    "strongest_connection": max_weight if edge_count > 0 else 0,
                    "weakest_connection": min_weight if edge_count > 0 and min_weight != float('inf') else 0,
                    "total_internal_edges": edge_count,
                    "sample_people": people_in_subgroup
                })
        
        # Sort by connection strength (descending)
        subgroups.sort(key=lambda x: x["avg_connection_strength"], reverse=True)
        
        return subgroups

    def get_subgraph_info(
        self, nodes: Set[int], feature_embeddings: Dict[str, np.ndarray] = None
    ) -> Dict:
        """Get detailed information about a subgraph including centroid insights and subgroups"""
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

        # Add subgroup analysis
        subgroups = self.find_subgroups_in_subgraph(nodes)
        if subgroups:
            result["subgroups"] = subgroups
            result["subgroup_summary"] = {
                "total_subgroups": len(subgroups),
                "strongest_subgroup_strength": subgroups[0]["avg_connection_strength"] if subgroups else 0,
                "largest_subgroup_size": max(sg["size"] for sg in subgroups) if subgroups else 0,
                "avg_subgroup_density": sum(sg["internal_density"] for sg in subgroups) / len(subgroups) if subgroups else 0
            }

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
        await self.create_graph(feature_embeddings)

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

        # Display subgroup analysis
        if "subgroups" in result:
            print(f"\n🔍 SUBGROUP ANALYSIS - Cohesive Clusters within Dense Subgraph:")
            print("=" * 70)
            summary = result["subgroup_summary"]
            print(f"Found {summary['total_subgroups']} cohesive subgroups")
            print(f"Strongest subgroup connection strength: {summary['strongest_subgroup_strength']:.3f}")
            print(f"Average subgroup density: {summary['avg_subgroup_density']:.3f}")
            
            for i, subgroup in enumerate(result["subgroups"][:5], 1):  # Show top 5 subgroups
                print(f"\n🔸 Subgroup {subgroup['subgroup_id']} ({subgroup['size']} people):")
                print(f"   Connection Strength: {subgroup['avg_connection_strength']:.3f}")
                print(f"   Internal Density: {subgroup['internal_density']:.3f}")
                print(f"   Strongest Connection: {subgroup['strongest_connection']:.3f}")
                print("   Sample Members:")
                for person in subgroup["sample_people"][:3]:
                    print(f"     • {person['name']} ({person['title']} at {person['company']})")

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

    async def create_graph_faiss(
        self, feature_embeddings: Dict[str, np.ndarray]
    ) -> nx.Graph:
        """Create graph using FAISS for maximum performance on large datasets"""
        
        top_k = int(os.getenv("FAISS_TOP_K", "50"))
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
                experience_complementarity = await self.causal_analyzer.calculate_experience_complementarity_fast(
                    self.extract_tags(self.df.iloc[person_i].get("Professional Identity - Experience Level", "")),
                    self.extract_tags(self.df.iloc[person_j].get("Professional Identity - Experience Level", ""))
                )
            else:
                experience_complementarity = 0.0

            # Business complementarity
            person1_business_tags = await self.extract_business_tags_for_person(
                self.df.iloc[person_i]
            )
            person2_business_tags = await self.extract_business_tags_for_person(
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
                await self.causal_analyzer.calculate_persona_complementarity_fast(
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
        "/home/ryan/PycharmProjects/match_engine/data/original.csv"
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
