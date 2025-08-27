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

from services.causal_relationship_analyzer import CausalRelationshipAnalyzer
from services.embedding_service import embedding_service
from services.faiss_similarity_engine import FAISSGraphMatcher, FAISSSimilarityEngine
from services.redis_cache import RedisEmbeddingCache
from services.semantic_person_deduplicator import SemanticPersonDeduplicator
from services.tag_extractor import tag_extractor

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
            "All Persona Titles",
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
                df[column].notna()
                & (df[column].astype(str).str.strip() != "")
                & (df[column].astype(str).str.strip() != "nan")
            )
            mask = mask & column_mask

            # Show which rows would be filtered by this column
            filtered_by_column = (~column_mask).sum()
            if filtered_by_column > 0:
                print(
                    f"   • {column}: {filtered_by_column} rows have empty/missing values"
                )

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

    def calculate_2plus2_score(
        self, 
        role_sim: float, exp_sim: float, role_comp: float, exp_comp: float,
        industry_sim: float, market_sim: float, offering_sim: float, persona_sim: float,
        business_comp: float,
        weights: List[float]
    ) -> float:
        """
        Calculate 2+2 architecture score with geometric mean within categories
        
        Args:
            Person similarities: role_sim, exp_sim
            Person complementarities: role_comp, exp_comp
            Business similarities: industry_sim, market_sim, offering_sim, persona_sim
            Business complementarity: business_comp (combined)
            weights: [person_sim_weight, person_comp_weight, business_sim_weight, business_comp_weight]
        
        Returns:
            Final score using 2+2 architecture
        """
        
        # Person dimension - geometric mean (role weighted higher than experience)
        if role_sim > 0 and exp_sim > 0:
            person_similarity = (role_sim ** 0.6) * (exp_sim ** 0.4)
        else:
            person_similarity = 0.0
            
        if role_comp > 0 and exp_comp > 0:
            person_complementarity = (role_comp ** 0.6) * (exp_comp ** 0.4)
        else:
            person_complementarity = 0.0
        
        # Business dimension - geometric mean (industry weighted highest)
        if all(x > 0 for x in [industry_sim, market_sim, offering_sim, persona_sim]):
            business_similarity = ((industry_sim ** 0.3) * (market_sim ** 0.25) * 
                                 (offering_sim ** 0.25) * (persona_sim ** 0.2))
        else:
            business_similarity = 0.0
        
        # Business complementarity - keep existing combined score
        business_complementarity = max(0.0, business_comp)
        
        # Final 2+2 linear combination
        final_score = (
            weights[0] * person_similarity + 
            weights[1] * person_complementarity +
            weights[2] * business_similarity + 
            weights[3] * business_complementarity
        )
        
        return max(0.0, min(1.0, final_score))  # Clamp to [0,1]

    async def get_tuned_2plus2_weights(self, user_prompt: str = None) -> List[float]:
        """
        Get 4 weights for 2+2 architecture using ChatGPT or defaults
        
        Args:
            user_prompt: User's intent description for weight tuning
            
        Returns:
            [person_sim_weight, person_comp_weight, business_sim_weight, business_comp_weight]
        """
        
        if not user_prompt or not user_prompt.strip():
            # Return balanced defaults
            return [0.25, 0.25, 0.25, 0.25]
        
        prompt = f"""User request: "{user_prompt}"

Provide exactly 4 weights that sum to 1.0 for professional networking scoring:

1. person_similarity_weight: How much SIMILAR roles and experience levels matter (0.0-1.0)
2. person_complementarity_weight: How much COMPLEMENTARY roles and experience levels matter (0.0-1.0)  
3. business_similarity_weight: How much SIMILAR industries, markets, offerings matter (0.0-1.0)
4. business_complementarity_weight: How much COMPLEMENTARY business contexts matter (0.0-1.0)

Guidelines by user intent:
- "Find AI executives/heads/CEOs" → [0.4, 0.1, 0.4, 0.1] (high similarity both dimensions)
- "Find partnership opportunities" → [0.2, 0.2, 0.1, 0.5] (high business complementarity) 
- "Find mentors/advisors" → [0.1, 0.4, 0.3, 0.2] (experience complementarity + domain similarity)
- "Find investors/funding" → [0.15, 0.15, 0.2, 0.5] (high business complementarity for investment thesis)
- "Find customers/clients" → [0.2, 0.1, 0.1, 0.6] (very high business complementarity)
- "General networking" → [0.25, 0.25, 0.25, 0.25] (balanced)

Respond with ONLY 4 numbers in this exact format: [0.25, 0.25, 0.25, 0.25]
Do not include any other text or explanation."""

        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50,
            )

            response_text = response.choices[0].message.content.strip()
            
            # Extract the list from the response
            import re
            match = re.search(r'\[(.*?)\]', response_text)
            if match:
                weights_str = match.group(1)
                weights = [float(w.strip()) for w in weights_str.split(',')]
                
                # Validate we got exactly 4 weights
                if len(weights) != 4:
                    raise ValueError(f"Expected 4 weights, got {len(weights)}")
                
                # Normalize to sum to 1.0
                total = sum(weights)
                if total > 0:
                    weights = [w/total for w in weights]
                else:
                    raise ValueError("All weights are zero")
                
                # Validate all weights are non-negative
                if any(w < 0 for w in weights):
                    raise ValueError("Negative weights not allowed")
                
                print(f"🎯 ChatGPT tuned weights for '{user_prompt}': {[f'{w:.3f}' for w in weights]}")
                return weights
                
            else:
                raise ValueError("Could not parse weights from ChatGPT response")
                
        except Exception as e:
            print(f"⚠️ ChatGPT weight tuning failed: {e}")
            print("Using balanced default weights")
            return [0.25, 0.25, 0.25, 0.25]

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

    async def create_graph(self, feature_embeddings: Dict[str, np.ndarray], user_prompt: str = None) -> nx.Graph:
        """Create graph using FAISS optimization for performance on large datasets"""
        
        use_faiss = os.getenv("USE_FAISS_OPTIMIZATION", "true").lower() == "true"
        
        if use_faiss:
            return await self.create_graph_faiss(feature_embeddings, user_prompt)
        else:
            return await self.create_graph_optimized(feature_embeddings, user_prompt)

    async def create_graph_optimized(
        self, feature_embeddings: Dict[str, np.ndarray], user_prompt: str = None
    ) -> nx.Graph:
        """Optimized graph creation with 2+2 architecture and ChatGPT weight tuning"""

        self.graph = nx.Graph()
        num_people = len(self.df)

        # Add nodes (people)
        for idx, row in self.df.iterrows():
            self.graph.add_node(
                idx, name=row["Person Name"], company=row["Person Company"]
            )

        # Load/build all complementarity matrices
        if not self.causal_analyzer.load_causal_graph_from_redis():
            print(
                "🔧 Business complementarity matrix not found. Building automatically..."
            )
            try:
                await self.causal_analyzer.build_causal_relationship_graph(
                    self.csv_path
                )
                print("✅ Business complementarity matrix built and cached")
            except Exception as e:
                print(f"⚠️ Error building business matrix: {e}")
                print("Using embedding-based complementarity fallback...")

        if not self.causal_analyzer.load_experience_matrix_from_redis():
            print(
                "🔧 Experience complementarity matrix not found. Building automatically..."
            )
            try:
                await self.causal_analyzer.build_experience_complementarity_matrix(
                    self.csv_path
                )
                print("✅ Experience complementarity matrix built and cached")
            except Exception as e:
                print(f"⚠️ Using embedding-based experience complementarity fallback")

        if not self.causal_analyzer.load_role_matrix_from_redis():
            print(
                "🔧 Role complementarity matrix not found. Building automatically..."
            )
            try:
                await self.causal_analyzer.build_role_complementarity_matrix(
                    self.csv_path
                )
                print("✅ Role complementarity matrix built and cached")
            except Exception as e:
                print(f"⚠️ Using embedding-based role complementarity fallback")

        # Get 2+2 architecture weights using ChatGPT or defaults
        weights = await self.get_tuned_2plus2_weights(user_prompt)
        
        print(f"📊 Using 2+2 architecture with weights:")
        print(f"   Person Similarity: {weights[0]:.3f}")
        print(f"   Person Complementarity: {weights[1]:.3f}")
        print(f"   Business Similarity: {weights[2]:.3f}")
        print(f"   Business Complementarity: {weights[3]:.3f}")
        print(f"   Total: {sum(weights):.3f}")

        # OPTIMIZATION 1: Precompute similarity matrices once
        print("⚡ Precomputing similarity matrices...")
        role_sim_matrix = None
        exp_sim_matrix = None
        persona_sim_matrix = None
        industry_sim_matrix = None
        market_sim_matrix = None
        offering_sim_matrix = None

        if "role_spec" in feature_embeddings:
            role_sim_matrix = cosine_similarity(feature_embeddings["role_spec"])

        if "experience" in feature_embeddings:
            exp_sim_matrix = cosine_similarity(feature_embeddings["experience"])

        if "personas" in feature_embeddings:
            persona_sim_matrix = cosine_similarity(feature_embeddings["personas"])

        if "industry" in feature_embeddings:
            industry_sim_matrix = cosine_similarity(feature_embeddings["industry"])

        if "market" in feature_embeddings:
            market_sim_matrix = cosine_similarity(feature_embeddings["market"])

        if "offering" in feature_embeddings:
            offering_sim_matrix = cosine_similarity(feature_embeddings["offering"])

        # OPTIMIZATION 2: Precompute all person tags once
        print("⚡ Precomputing person tags...")
        person_tags_cache = {}
        for idx, row in self.df.iterrows():
            person_tags_cache[idx] = {
                "experience": self.extract_tags(
                    row.get("Professional Identity - Experience Level", "")
                ),
                "personas": self.extract_tags(row.get("All Persona Titles", "")),
                "business": None,  # Will compute async below
            }

        # Precompute business tags (async)
        print("⚡ Precomputing business tags...")
        for idx, row in self.df.iterrows():
            person_tags_cache[idx]["business"] = (
                await self.extract_business_tags_for_person(row)
            )

        # OPTIMIZATION 3: Vectorized edge calculation with early termination
        print(
            f"⚡ Computing {num_people * (num_people - 1) // 2} pairwise similarities..."
        )
        edges_added = 0

        for i in range(num_people):
            for j in range(i + 1, num_people):

                # Person Similarities
                role_similarity = role_sim_matrix[i][j] if role_sim_matrix is not None else 0.0
                experience_similarity = exp_sim_matrix[i][j] if exp_sim_matrix is not None else 0.0
                
                # Business Similarities 
                industry_similarity = industry_sim_matrix[i][j] if industry_sim_matrix is not None else 0.0
                market_similarity = market_sim_matrix[i][j] if market_sim_matrix is not None else 0.0
                offering_similarity = offering_sim_matrix[i][j] if offering_sim_matrix is not None else 0.0
                persona_similarity = persona_sim_matrix[i][j] if persona_sim_matrix is not None else 0.0

                # Person Complementarities
                role_complementarity = await self.causal_analyzer.calculate_role_complementarity_fast(
                    person_tags_cache[i]["roles"],
                    person_tags_cache[j]["roles"],
                )
                
                experience_complementarity = await self.causal_analyzer.calculate_experience_complementarity_fast(
                    person_tags_cache[i]["experience"],
                    person_tags_cache[j]["experience"],
                )

                # Business Complementarity (combined)
                business_complementarity = self.causal_analyzer.calculate_business_complementarity_fast(
                    person_tags_cache[i]["business"],
                    person_tags_cache[j]["business"],
                )

                # NEW: 2+2 ARCHITECTURE SCORING
                score_2plus2 = self.calculate_2plus2_score(
                    role_similarity, experience_similarity, role_complementarity, experience_complementarity,
                    industry_similarity, market_similarity, offering_similarity, persona_similarity,
                    business_complementarity,
                    weights
                )

                # Only add edges above threshold
                if score_2plus2 > 0.1:
                    self.graph.add_edge(
                        i,
                        j,
                        weight=score_2plus2,
                        # Store all individual components for debugging
                        role_similarity=role_similarity,
                        experience_similarity=experience_similarity,
                        role_complementarity=role_complementarity,
                        experience_complementarity=experience_complementarity,
                        industry_similarity=industry_similarity,
                        market_similarity=market_similarity,
                        offering_similarity=offering_similarity,
                        persona_similarity=persona_similarity,
                        business_complementarity=business_complementarity,
                    )
                    edges_added += 1

        print(f"✅ Created 2+2 architecture graph:")
        print(f"   Nodes: {self.graph.number_of_nodes()}")
        print(f"   Edges: {edges_added}")
        print(
            f"   Density: {edges_added / (num_people * (num_people - 1) // 2) * 100:.1f}%"
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

    def find_subgroups_in_subgraph(
        self, nodes: Set[int], min_subgroup_size: int = 3
    ) -> List[Dict]:
        """Find cohesive subgroups within the dense subgraph using edge weights"""
        if len(nodes) < min_subgroup_size * 2:
            return []

        subgraph = self.graph.subgraph(nodes)

        # Use Louvain community detection for clustering
        try:
            communities = nx.community.louvain_communities(
                subgraph, weight="weight", resolution=1.2
            )
        except:
            # Fallback to greedy modularity if Louvain fails
            communities = nx.community.greedy_modularity_communities(
                subgraph, weight="weight"
            )

        subgroups = []

        for i, community in enumerate(communities):
            if len(community) >= min_subgroup_size:
                # Calculate internal density and average weight
                community_subgraph = subgraph.subgraph(community)

                total_weight = 0
                edge_count = 0
                max_weight = 0
                min_weight = float("inf")

                for u, v, data in community_subgraph.edges(data=True):
                    weight = data.get("weight", 0)
                    total_weight += weight
                    edge_count += 1
                    max_weight = max(max_weight, weight)
                    min_weight = min(min_weight, weight)

                avg_internal_weight = total_weight / edge_count if edge_count > 0 else 0

                # Calculate density specifically for this subgroup
                possible_edges = len(community) * (len(community) - 1) / 2
                internal_density = (
                    edge_count / possible_edges if possible_edges > 0 else 0
                )

                # Get representative people from this subgroup
                people_in_subgroup = []
                for node in list(community)[:5]:  # Show up to 5 people
                    row = self.df.iloc[node]
                    people_in_subgroup.append(
                        {
                            "name": row["Person Name"],
                            "title": row["Person Title"],
                            "company": row["Person Company"],
                        }
                    )

                subgroups.append(
                    {
                        "subgroup_id": i + 1,
                        "size": len(community),
                        "internal_density": internal_density,
                        "avg_connection_strength": avg_internal_weight,
                        "strongest_connection": max_weight if edge_count > 0 else 0,
                        "weakest_connection": (
                            min_weight
                            if edge_count > 0 and min_weight != float("inf")
                            else 0
                        ),
                        "total_internal_edges": edge_count,
                        "sample_people": people_in_subgroup,
                    }
                )

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
                "strongest_subgroup_strength": (
                    subgroups[0]["avg_connection_strength"] if subgroups else 0
                ),
                "largest_subgroup_size": (
                    max(sg["size"] for sg in subgroups) if subgroups else 0
                ),
                "avg_subgroup_density": (
                    sum(sg["internal_density"] for sg in subgroups) / len(subgroups)
                    if subgroups
                    else 0
                ),
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
            print(
                f"Strongest subgroup connection strength: {summary['strongest_subgroup_strength']:.3f}"
            )
            print(f"Average subgroup density: {summary['avg_subgroup_density']:.3f}")

            for i, subgroup in enumerate(
                result["subgroups"][:5], 1
            ):  # Show top 5 subgroups
                print(
                    f"\n🔸 Subgroup {subgroup['subgroup_id']} ({subgroup['size']} people):"
                )
                print(
                    f"   Connection Strength: {subgroup['avg_connection_strength']:.3f}"
                )
                print(f"   Internal Density: {subgroup['internal_density']:.3f}")
                print(
                    f"   Strongest Connection: {subgroup['strongest_connection']:.3f}"
                )
                print("   Sample Members:")
                for person in subgroup["sample_people"][:3]:
                    print(
                        f"     • {person['name']} ({person['title']} at {person['company']})"
                    )

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
        self, feature_embeddings: Dict[str, np.ndarray], user_prompt: str = None
    ) -> nx.Graph:
        """Create graph using FAISS with 2+2 architecture for maximum performance"""

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

        # Get 2+2 architecture weights using ChatGPT or defaults
        weights = await self.get_tuned_2plus2_weights(user_prompt)
        
        print(f"📊 Using 2+2 FAISS architecture with weights:")
        print(f"   Person Similarity: {weights[0]:.3f}")
        print(f"   Person Complementarity: {weights[1]:.3f}")
        print(f"   Business Similarity: {weights[2]:.3f}")
        print(f"   Business Complementarity: {weights[3]:.3f}")
        print(f"   Total: {sum(weights):.3f}")

        faiss_matcher = FAISSGraphMatcher(faiss_engine)

        role_similarities = faiss_matcher.get_similarity_pairs(
            "role_spec", top_k=top_k, min_similarity=0.05
        )
        exp_similarities = faiss_matcher.get_similarity_pairs(
            "experience", top_k=top_k, min_similarity=0.05
        )
        persona_similarities = faiss_matcher.get_similarity_pairs(
            "personas", top_k=top_k, min_similarity=0.05
        )
        industry_similarities = faiss_matcher.get_similarity_pairs(
            "industry", top_k=top_k, min_similarity=0.05
        )
        market_similarities = faiss_matcher.get_similarity_pairs(
            "market", top_k=top_k, min_similarity=0.05
        )
        offering_similarities = faiss_matcher.get_similarity_pairs(
            "offering", top_k=top_k, min_similarity=0.05
        )

        all_candidate_pairs = (
            set(role_similarities.keys())
            | set(exp_similarities.keys())
            | set(persona_similarities.keys())
            | set(industry_similarities.keys())
            | set(market_similarities.keys())
            | set(offering_similarities.keys())
        )

        print(
            f"⚡ FAISS found {len(all_candidate_pairs)} candidate pairs from all similarity components"
        )

        # Load causal/complementarity analyzers
        if not self.causal_analyzer.load_causal_graph_from_redis():
            print(
                "⚠️ No business complementarity matrix found. Building automatically..."
            )
            try:
                await self.causal_analyzer.build_causal_relationship_graph(
                    self.csv_path
                )
                print("✅ Business complementarity matrix built and cached")
            except Exception as e:
                print(
                    f"⚠️ Error building business matrix, using similarity-only fallback"
                )
                enhanced_edges = {}
                for person_i, person_j in all_candidate_pairs:
                    role_sim = role_similarities.get((person_i, person_j), 0.0)
                    exp_sim = exp_similarities.get((person_i, person_j), 0.0)
                    persona_sim = persona_similarities.get((person_i, person_j), 0.0)

                    similarity_only = (
                        role_weight * role_sim
                        + exp_sim_weight * exp_sim
                        + persona_sim_weight * persona_sim
                    )

                    if similarity_only > 0.1:
                        enhanced_edges[(person_i, person_j)] = similarity_only

                for (person_i, person_j), weight in enhanced_edges.items():
                    self.graph.add_edge(person_i, person_j, weight=weight)

                print(
                    f"Created FAISS similarity-only graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
                )
                return self.graph

        if not self.causal_analyzer.load_experience_matrix_from_redis():
            print(
                "🔧 Experience complementarity matrix not found. Building automatically..."
            )
            try:
                await self.causal_analyzer.build_experience_complementarity_matrix(
                    self.csv_path
                )
                print("✅ Experience complementarity matrix built and cached")
            except Exception as e:
                print(f"⚠️ Using embedding-based experience complementarity fallback")

        if not self.causal_analyzer.load_role_matrix_from_redis():
            print(
                "🔧 Role complementarity matrix not found. Building automatically..."
            )
            try:
                await self.causal_analyzer.build_role_complementarity_matrix(
                    self.csv_path
                )
                print("✅ Role complementarity matrix built and cached")
            except Exception as e:
                print(f"⚠️ Using embedding-based role complementarity fallback")

        print("⚡ Precomputing person tags for complementarity...")
        person_tags_cache = {}
        for idx, row in self.df.iterrows():
            person_tags_cache[idx] = {
                "experience": self.extract_tags(
                    row.get("Professional Identity - Experience Level", "")
                ),
                "roles": self.extract_tags(
                    row.get("Professional Identity - Role Specification", "")
                ),
                "personas": self.extract_tags(row.get("All Persona Titles", "")),
                "business": await self.extract_business_tags_for_person(row),
            }

        print("⚡ Applying 2+2 architecture scoring to all candidate pairs...")
        enhanced_edges = {}

        for person_i, person_j in all_candidate_pairs:

            # Person Similarities
            role_sim = role_similarities.get((person_i, person_j), 0.0)
            exp_sim = exp_similarities.get((person_i, person_j), 0.0)
            
            # Business Similarities
            industry_sim = industry_similarities.get((person_i, person_j), 0.0)
            market_sim = market_similarities.get((person_i, person_j), 0.0)
            offering_sim = offering_similarities.get((person_i, person_j), 0.0)
            persona_sim = persona_similarities.get((person_i, person_j), 0.0)

            # Person Complementarities
            role_complementarity = await self.causal_analyzer.calculate_role_complementarity_fast(
                person_tags_cache[person_i]["roles"],
                person_tags_cache[person_j]["roles"],
            )
            
            experience_complementarity = await self.causal_analyzer.calculate_experience_complementarity_fast(
                person_tags_cache[person_i]["experience"],
                person_tags_cache[person_j]["experience"],
            )

            # Business Complementarity (combined)
            business_complementarity = self.causal_analyzer.calculate_business_complementarity_fast(
                person_tags_cache[person_i]["business"],
                person_tags_cache[person_j]["business"],
            )

            # NEW: 2+2 ARCHITECTURE SCORING
            score_2plus2 = self.calculate_2plus2_score(
                role_sim, exp_sim, role_complementarity, experience_complementarity,
                industry_sim, market_sim, offering_sim, persona_sim,
                business_complementarity,
                weights
            )

            if score_2plus2 > 0.1:
                enhanced_edges[(person_i, person_j)] = score_2plus2

        for (person_i, person_j), weight in enhanced_edges.items():
            self.graph.add_edge(person_i, person_j, weight=weight)

        print(
            f"✅ Created 2+2 FAISS-optimized graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
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
