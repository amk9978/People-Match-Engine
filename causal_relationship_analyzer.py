#!/usr/bin/env python3

import asyncio
import json
import os
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from redis_cache import RedisEmbeddingCache

load_dotenv()


class CausalRelationshipAnalyzer:
    """Analyze causal/strategic relationships between business tags using ChatGPT"""

    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cache = RedisEmbeddingCache()
        self.causal_graph = {}
        self.CAUSAL_GRAPH_KEY = "causal_graph_complete"

    def extract_business_tags_from_dataset(self, csv_path: str) -> Dict[str, Set[str]]:
        """Extract all unique business tags from dataset"""
        df = pd.read_csv(csv_path)

        business_columns = {
            "industry": "Company Identity - Industry Classification",
            "market": "Company Market - Market Traction",
            "offering": "Company Offering - Value Proposition",
        }

        business_tags = {category: set() for category in business_columns.keys()}

        for category, column_name in business_columns.items():
            for _, row in df.iterrows():
                if pd.notna(row[column_name]):
                    # Split by pipe and clean
                    tags = [tag.strip() for tag in str(row[column_name]).split("|")]
                    business_tags[category].update(tag for tag in tags if tag)

        print(f"Extracted business tags:")
        for category, tags in business_tags.items():
            print(f"  {category}: {len(tags)} unique tags")

        return business_tags

    async def get_causal_relationships_for_tag(
        self, target_tag: str, comparison_tags: List[str], category: str
    ) -> Dict[str, float]:
        """Get causal relationship scores for one tag against a list of others"""

        # Check cache first
        cache_key = f"causal_{category}_{target_tag}_vs_{len(comparison_tags)}"
        cached_result = self.cache.get(cache_key)
        if cached_result and isinstance(cached_result, str):
            return json.loads(cached_result)

        # Prepare ChatGPT prompt
        comparison_list = "\n".join([f"- {tag}" for tag in comparison_tags])

        prompt = f"""
You are a business strategy expert analyzing complementary relationships between {category} characteristics.

TARGET: {target_tag}

Rate the STRATEGIC VALUE of business connections between "{target_tag}" and each of these other {category} characteristics:

{comparison_list}

Scoring criteria (0.0 to 1.0):
- 0.9-1.0: Highly complementary, creates significant strategic value (e.g., "Early-stage" + "Growth-stage" for mentorship/investment)
- 0.7-0.8: Strong complementary value (e.g., "B2B SaaS" + "Enterprise Hardware" for integration opportunities)  
- 0.5-0.6: Moderate complementary value (some synergies possible)
- 0.3-0.4: Limited complementary value (different but not particularly synergistic)
- 0.1-0.2: Minimal complementary value (too similar or unrelated)
- 0.0: No strategic value (identical or conflicting)

IMPORTANT: Rate COMPLEMENTARY VALUE, not similarity. Different industries/stages often create MORE strategic value.

CRITICAL: You MUST respond with ONLY a valid JSON object. No explanations, no markdown, no text before or after. Start your response with {{ and end with }}. Example format:
{{"Artificial Intelligence": 0.85, "Manufacturing": 0.92, "Healthcare": 0.76}}

Your response:"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )

            result_text = response.choices[0].message.content.strip()

            # Enhanced JSON parsing with multiple fallback strategies
            causal_scores = self._parse_chatgpt_response(result_text, comparison_tags)

            # Cache the result
            self.cache.set(cache_key, json.dumps(causal_scores))

            print(
                f"  ✓ Got causal scores for {target_tag} vs {len(causal_scores)} tags"
            )
            return causal_scores

        except Exception as e:
            print(f"  ✗ Error getting causal scores for {target_tag}: {e}")
            # Return default moderate scores
            return {tag: 0.5 for tag in comparison_tags}

    def _parse_chatgpt_response(
        self, result_text: str, comparison_tags: List[str]
    ) -> Dict[str, float]:
        """Parse ChatGPT response with multiple fallback strategies"""

        # Strategy 1: Try direct JSON parsing
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Remove markdown formatting
        try:
            if "```json" in result_text:
                json_part = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                json_part = result_text.split("```")[1].split("```")[0].strip()
            else:
                json_part = result_text

            return json.loads(json_part)
        except (json.JSONDecodeError, IndexError):
            pass

        # Strategy 3: Find JSON-like content with regex
        try:
            import re

            json_match = re.search(r"\{[^{}]*\}", result_text, re.DOTALL)
            if json_match:
                json_content = json_match.group(0)
                return json.loads(json_content)
        except (json.JSONDecodeError, AttributeError):
            pass

        # Strategy 4: Extract key-value pairs manually
        try:
            import re

            # Look for "tag": score or "tag": score patterns
            pattern = r'"([^"]+)":\s*([0-9.]+)'
            matches = re.findall(pattern, result_text)

            if matches:
                result = {}
                for tag_name, score_str in matches:
                    try:
                        result[tag_name] = float(score_str)
                    except ValueError:
                        continue

                # Validate we got scores for most tags
                if len(result) >= len(comparison_tags) * 0.5:  # At least 50% success
                    # Fill in missing tags with default score
                    for tag in comparison_tags:
                        if tag not in result:
                            result[tag] = 0.5
                    return result
        except Exception:
            pass

        # Final fallback: Return default scores with warning
        print(f"    Warning: Could not parse ChatGPT response, using default scores")
        print(f"    Response was: {result_text[:200]}...")
        return {tag: 0.5 for tag in comparison_tags}

    async def build_causal_relationship_graph(
        self,
        csv_path: str,
        max_concurrent_requests: int = 5,
        rate_limit_delay: float = 0.2,
    ) -> Dict[str, Dict[str, float]]:
        """Build complete causal relationship graph with optimized parallel processing"""

        print(
            "Building causal relationship graph with optimized parallel processing..."
        )

        # Extract all business tags
        business_tags = self.extract_business_tags_from_dataset(csv_path)

        causal_graph = {}

        # Process each category with controlled concurrency
        for category, tags in business_tags.items():
            print(f"\nProcessing {category} tags ({len(tags)} total)...")

            category_graph = {}
            tags_list = list(tags)

            # Create all tasks first
            all_tasks = []
            for target_tag in tags_list:
                comparison_tags = [tag for tag in tags_list if tag != target_tag]
                task = self.get_causal_relationships_for_tag(
                    target_tag, comparison_tags, category
                )
                all_tasks.append((target_tag, task))

            print(f"  Created {len(all_tasks)} tasks for parallel processing...")

            # Process with semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent_requests)

            async def rate_limited_task(target_tag, task):
                async with semaphore:
                    try:
                        result = await task
                        # Small delay between requests to respect rate limits
                        await asyncio.sleep(rate_limit_delay)
                        return target_tag, result
                    except Exception as e:
                        print(f"    Error for {target_tag}: {e}")
                        return target_tag, None

            # Execute all tasks with controlled concurrency
            print(
                f"  Executing with max {max_concurrent_requests} concurrent requests..."
            )

            completed_tasks = await asyncio.gather(
                *[rate_limited_task(tag, task) for tag, task in all_tasks],
                return_exceptions=True,
            )

            # Process results
            successful_count = 0
            for result in completed_tasks:
                if isinstance(result, Exception):
                    print(f"    Task failed: {result}")
                    continue

                target_tag, causal_scores = result
                if causal_scores is not None:
                    category_graph[target_tag] = causal_scores
                    successful_count += 1

                    # Progress indicator
                    if successful_count % 5 == 0:
                        print(
                            f"    ✓ Completed {successful_count}/{len(all_tasks)} tags"
                        )

            causal_graph[category] = category_graph
            print(
                f"  ✓ Completed {category}: {successful_count}/{len(all_tasks)} tags processed successfully"
            )

        self.causal_graph = causal_graph

        # Cache the complete causal graph in Redis
        self.cache_causal_graph_to_redis(causal_graph)

        return causal_graph

    def get_causal_score(self, tag1: str, tag2: str, category: str) -> float:
        """Get causal relationship score between two tags"""
        if category not in self.causal_graph:
            return 0.5  # Default moderate score

        category_graph = self.causal_graph[category]

        # Try both directions
        if tag1 in category_graph and tag2 in category_graph[tag1]:
            return category_graph[tag1][tag2]
        elif tag2 in category_graph and tag1 in category_graph[tag2]:
            return category_graph[tag2][tag1]
        else:
            return 0.5  # Default moderate score

    def calculate_business_complementarity(
        self, person1_tags: Dict[str, List[str]], person2_tags: Dict[str, List[str]]
    ) -> float:
        """Calculate overall business complementarity score between two people"""

        total_score = 0.0
        total_comparisons = 0

        # Weights for different business aspects
        category_weights = {"industry": 0.4, "market": 0.35, "offering": 0.25}

        for category, weight in category_weights.items():
            if category in person1_tags and category in person2_tags:
                category_scores = []

                # Compare all tag pairs in this category
                for tag1 in person1_tags[category]:
                    for tag2 in person2_tags[category]:
                        score = self.get_causal_score(tag1, tag2, category)
                        category_scores.append(score)

                if category_scores:
                    # Use max score for this category (best complementary match)
                    category_best = max(category_scores)
                    total_score += weight * category_best
                    total_comparisons += weight

        return total_score / total_comparisons if total_comparisons > 0 else 0.5

    def cache_causal_graph_to_redis(
        self, causal_graph: Dict[str, Dict[str, Dict[str, float]]]
    ):
        """Cache the complete causal graph to Redis as a flattened adjacency matrix"""
        print("Caching causal relationship graph to Redis...")

        # Flatten the graph into individual key-value pairs for fast lookup
        cached_count = 0

        for category, category_graph in causal_graph.items():
            for tag1, relationships in category_graph.items():
                for tag2, score in relationships.items():
                    # Create unique Redis key for each pair
                    redis_key = f"causal:{category}:{tag1}:{tag2}"
                    self.cache.set(redis_key, str(score))
                    cached_count += 1

        # Also cache the complete graph as JSON for backup
        complete_graph_json = json.dumps(causal_graph)
        self.cache.set(self.CAUSAL_GRAPH_KEY, complete_graph_json)

        print(f"  ✓ Cached {cached_count} causal relationships to Redis")
        print(f"  ✓ Cached complete graph backup to Redis")

    def load_causal_graph_from_redis(self) -> bool:
        """Load the complete causal graph from Redis"""
        try:
            # Try to load complete graph first
            cached_graph = self.cache.get(self.CAUSAL_GRAPH_KEY)
            if cached_graph and isinstance(cached_graph, str):
                self.causal_graph = json.loads(cached_graph)
                print(f"✓ Loaded complete causal graph from Redis")
                return True
            else:
                print("No complete causal graph found in Redis")
                return False

        except Exception as e:
            print(f"Error loading causal graph from Redis: {e}")
            return False

    def get_causal_score_from_redis(self, tag1: str, tag2: str, category: str) -> float:
        """Fast lookup of causal score directly from Redis"""

        # Try both directions
        redis_key1 = f"causal:{category}:{tag1}:{tag2}"
        redis_key2 = f"causal:{category}:{tag2}:{tag1}"

        score = self.cache.get(redis_key1)
        if score:
            return float(score)

        score = self.cache.get(redis_key2)
        if score:
            return float(score)

        return 0.5  # Default moderate score if not found

    def calculate_business_complementarity_fast(
        self, person1_tags: Dict[str, List[str]], person2_tags: Dict[str, List[str]]
    ) -> float:
        """Fast calculation using Redis-cached causal scores"""

        total_score = 0.0
        total_comparisons = 0

        # Weights for different business aspects
        category_weights = {"industry": 0.4, "market": 0.35, "offering": 0.25}

        for category, weight in category_weights.items():
            if category in person1_tags and category in person2_tags:
                category_scores = []

                # Compare all tag pairs in this category using Redis lookup
                for tag1 in person1_tags[category]:
                    for tag2 in person2_tags[category]:
                        score = self.get_causal_score_from_redis(tag1, tag2, category)
                        category_scores.append(score)

                if category_scores:
                    # Use max score for this category (best complementary match)
                    category_best = max(category_scores)
                    total_score += weight * category_best
                    total_comparisons += weight

        return total_score / total_comparisons if total_comparisons > 0 else 0.5

    def save_causal_graph(self, output_path: str = "causal_relationship_graph.json"):
        """Save causal relationship graph to file"""
        with open(output_path, "w") as f:
            json.dump(self.causal_graph, f, indent=2)
        print(f"Causal relationship graph saved to {output_path}")

    def load_causal_graph(self, input_path: str = "causal_relationship_graph.json"):
        """Load causal relationship graph from file"""
        try:
            with open(input_path, "r") as f:
                self.causal_graph = json.load(f)
            print(f"Causal relationship graph loaded from {input_path}")
            return True
        except FileNotFoundError:
            print(f"Causal graph file not found: {input_path}")
            return False

    def print_causal_graph_summary(self):
        """Print summary of causal relationship graph"""
        if not self.causal_graph:
            print("No causal graph loaded")
            return

        print("\n" + "=" * 60)
        print("CAUSAL RELATIONSHIP GRAPH SUMMARY")
        print("=" * 60)

        for category, category_graph in self.causal_graph.items():
            print(f"\n📊 {category.upper()}:")
            print(f"  Total tags: {len(category_graph)}")

            # Find highest scoring relationships
            all_scores = []
            high_value_pairs = []

            for tag1, relationships in category_graph.items():
                for tag2, score in relationships.items():
                    all_scores.append(score)
                    if score >= 0.8:
                        high_value_pairs.append((tag1, tag2, score))

            if all_scores:
                print(
                    f"  Average complementarity: {sum(all_scores)/len(all_scores):.3f}"
                )
                print(f"  Max complementarity: {max(all_scores):.3f}")

                # Show top complementary pairs
                high_value_pairs.sort(key=lambda x: x[2], reverse=True)
                if high_value_pairs:
                    print(f"  Top complementary pairs:")
                    for tag1, tag2, score in high_value_pairs[:3]:
                        print(f"    • {tag1} ↔ {tag2}: {score:.3f}")

    def get_causal_graph_stats(self) -> Dict[str, any]:
        """Get statistics about the cached causal graph"""
        if not self.load_causal_graph_from_redis():
            return {"error": "No causal graph found in Redis"}

        stats = {}
        total_relationships = 0

        for category, category_graph in self.causal_graph.items():
            category_stats = {
                "unique_tags": len(category_graph),
                "total_relationships": 0,
                "avg_score": 0.0,
                "high_value_relationships": 0,  # Score >= 0.8
                "low_value_relationships": 0,  # Score <= 0.2
            }

            all_scores = []
            for tag, relationships in category_graph.items():
                category_stats["total_relationships"] += len(relationships)
                for score in relationships.values():
                    all_scores.append(score)
                    if score >= 0.8:
                        category_stats["high_value_relationships"] += 1
                    elif score <= 0.2:
                        category_stats["low_value_relationships"] += 1

            if all_scores:
                category_stats["avg_score"] = sum(all_scores) / len(all_scores)
                category_stats["min_score"] = min(all_scores)
                category_stats["max_score"] = max(all_scores)

            stats[category] = category_stats
            total_relationships += category_stats["total_relationships"]

        stats["overall"] = {
            "total_categories": len(self.causal_graph),
            "total_relationships": total_relationships,
            "graph_density": (
                total_relationships
                / (sum(len(cg) for cg in self.causal_graph.values()) ** 2)
                if self.causal_graph
                else 0
            ),
        }

        return stats

    def extract_persona_tags_from_dataset(self, csv_path: str) -> Set[str]:
        """Extract all unique persona tags from dataset"""
        df = pd.read_csv(csv_path)

        persona_tags = set()

        # Personas column
        persona_column = "All Persona Titles"
        if persona_column in df.columns:
            for value in df[persona_column].dropna():
                # Split by semicolon for personas
                tags = [tag.strip() for tag in str(value).split(";")]
                persona_tags.update(tag for tag in tags if tag)

        print(f"Extracted {len(persona_tags)} unique persona tags")
        return persona_tags

    async def analyze_persona_complementarity(
        self, persona1: str, persona2: str
    ) -> float:
        """Analyze complementarity between two persona tags using ChatGPT"""

        cache_key = f"persona_complementarity:{sorted([persona1, persona2])}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return float(cached_result)

        prompt = f"""You are analyzing professional persona complementarity for networking/partnership purposes.

PERSONA 1: {persona1}
PERSONA 2: {persona2}

Rate how well these two personas would complement each other in a professional context (0.0-1.0):

Consider:
- Strategic partnership potential (different strengths that combine well)
- Knowledge/skill complementarity (they can learn from each other)
- Network value (connecting them creates mutual benefit)
- Collaboration potential (they would work well together)

Examples:
- "Product Manager" + "Software Engineer" = 0.9 (perfect collaboration)
- "CEO" + "CTO" = 0.8 (strategic leadership combo)
- "Sales Manager" + "Marketing Director" = 0.7 (business development synergy)
- "Data Scientist" + "Business Analyst" = 0.8 (analytical complementarity)
- "CEO" + "CEO" = 0.3 (similar roles, less complementary)
- "Junior Developer" + "Senior Developer" = 0.6 (mentorship value)

Respond with ONLY a number between 0.0 and 1.0"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10,
            )

            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            score = max(0.0, min(1.0, score))  # Clamp to [0.0, 1.0]

            # Cache the result
            self.cache.set(cache_key, score)
            return score

        except Exception as e:
            print(f"Error analyzing persona complementarity {persona1}-{persona2}: {e}")
            return 0.5  # Neutral fallback

    async def build_persona_complementarity_matrix(
        self, csv_path: str
    ) -> Dict[str, Dict[str, float]]:
        """Build persona complementarity adjacency matrix using ChatGPT"""

        # Check if already cached
        cache_key = "persona_complementarity_matrix"
        cached_matrix = self.cache.get(cache_key)
        if cached_matrix:
            try:
                return json.loads(cached_matrix)
            except json.JSONDecodeError:
                pass

        print("Building persona complementarity matrix...")

        # Extract all unique persona tags
        persona_tags = self.extract_persona_tags_from_dataset(csv_path)
        persona_list = list(persona_tags)

        print(f"Analyzing complementarity for {len(persona_list)} personas...")

        # Create all pairs for analysis
        all_tasks = []
        for i, persona1 in enumerate(persona_list):
            for j, persona2 in enumerate(persona_list):
                if i <= j:  # Only analyze each pair once (symmetric matrix)
                    task = self.analyze_persona_complementarity(persona1, persona2)
                    all_tasks.append(((persona1, persona2), task))

        print(f"Total persona pairs to analyze: {len(all_tasks)}")

        # Rate limiting
        max_concurrent_requests = 5
        rate_limit_delay = 0.5
        semaphore = asyncio.Semaphore(max_concurrent_requests)

        async def rate_limited_task(pair_info, task):
            async with semaphore:
                try:
                    result = await task
                    await asyncio.sleep(rate_limit_delay)
                    return pair_info, result
                except Exception as e:
                    print(f"    Error for {pair_info}: {e}")
                    return pair_info, 0.5

        # Execute all tasks
        print(f"  Executing with max {max_concurrent_requests} concurrent requests...")
        completed_tasks = await asyncio.gather(
            *[rate_limited_task(pair_info, task) for pair_info, task in all_tasks],
            return_exceptions=True,
        )

        # Build matrix
        persona_matrix = {}
        for persona in persona_list:
            persona_matrix[persona] = {}

        successful_count = 0
        for result in completed_tasks:
            if isinstance(result, Exception):
                continue

            (persona1, persona2), score = result
            if score is not None:
                # Symmetric matrix
                persona_matrix[persona1][persona2] = score
                persona_matrix[persona2][persona1] = score
                successful_count += 1

        print(
            f"Successfully analyzed {successful_count}/{len(all_tasks)} persona pairs"
        )

        # Cache the result
        self.cache.set(cache_key, json.dumps(persona_matrix))

        return persona_matrix

    async def calculate_persona_complementarity_fast(
        self, person1_personas: List[str], person2_personas: List[str]
    ) -> float:
        """Fast calculation of persona complementarity using cached matrix"""

        if not hasattr(self, "persona_matrix") or not self.persona_matrix:
            # Try to load from cache first
            if self.load_persona_matrix_from_redis():
                pass  # Successfully loaded
            else:
                # No matrix found - build it automatically
                if not hasattr(self, "_building_matrix"):
                    print("🔧 Persona matrix not found. Building automatically...")
                    self._building_matrix = True
                    
                    # Get all personas from the current calculation
                    all_personas = set(person1_personas + person2_personas)
                    
                    # Build matrix just for personas we've seen so far (incremental)
                    await self.build_incremental_persona_matrix(all_personas)
                else:
                    # Already building, use fallback
                    return 0.5

        if not person1_personas or not person2_personas:
            return 0.0

        total_score = 0.0
        pair_count = 0

        for persona1 in person1_personas:
            for persona2 in person2_personas:
                if (
                    persona1 in self.persona_matrix
                    and persona2 in self.persona_matrix[persona1]
                ):
                    total_score += self.persona_matrix[persona1][persona2]
                    pair_count += 1

        return total_score / pair_count if pair_count > 0 else 0.0

    def load_persona_matrix_from_redis(self) -> bool:
        """Load persona complementarity matrix from Redis cache"""
        cached_matrix = self.cache.get("persona_complementarity_matrix")
        if cached_matrix:
            try:
                self.persona_matrix = json.loads(cached_matrix)
                return True
            except json.JSONDecodeError:
                return False
        return False

    async def build_incremental_persona_matrix(self, new_personas: set) -> None:
        """Build matrix incrementally for new personas encountered"""
        
        # Load existing matrix or start fresh
        if not hasattr(self, "persona_matrix"):
            self.persona_matrix = {}
        
        # Get personas that need to be added
        existing_personas = set(self.persona_matrix.keys()) if self.persona_matrix else set()
        personas_to_add = new_personas - existing_personas
        all_personas = existing_personas | new_personas
        
        if not personas_to_add:
            return  # Nothing new to add
        
        print(f"📈 Adding {len(personas_to_add)} new personas to matrix (total: {len(all_personas)})")
        
        # Build relationships for new personas
        for new_persona in personas_to_add:
            if new_persona not in self.persona_matrix:
                self.persona_matrix[new_persona] = {}
            
            # Calculate relationships with ALL personas (existing + new)
            for other_persona in all_personas:
                if new_persona != other_persona:
                    if other_persona not in self.persona_matrix:
                        self.persona_matrix[other_persona] = {}
                    
                    # Only calculate if not already exists
                    if other_persona not in self.persona_matrix[new_persona]:
                        try:
                            score = await self.calculate_persona_complementarity_chatgpt(
                                new_persona, other_persona
                            )
                            # Store symmetrically
                            self.persona_matrix[new_persona][other_persona] = score
                            self.persona_matrix[other_persona][new_persona] = score
                        except Exception as e:
                            print(f"Error calculating {new_persona} <-> {other_persona}: {e}")
                            # Use embedding similarity as fallback
                            score = await self.calculate_embedding_similarity_fallback(
                                new_persona, other_persona
                            )
                            self.persona_matrix[new_persona][other_persona] = score
                            self.persona_matrix[other_persona][new_persona] = score
        
        # Save updated matrix to Redis
        self.cache.set("persona_complementarity_matrix", json.dumps(self.persona_matrix))
        print(f"💾 Updated persona matrix cached with {len(self.persona_matrix)} personas")

    async def calculate_embedding_similarity_fallback(
        self, persona1: str, persona2: str
    ) -> float:
        """Use embedding similarity as fallback for complementarity calculation"""
        try:
            from embedding_service import embedding_service
            
            embeddings = await embedding_service.get_batch_embeddings([persona1, persona2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            # Convert similarity to complementarity (inverse relationship)
            complementarity = 1.0 - similarity
            return max(0.0, min(1.0, complementarity))  # Clamp to [0,1]
            
        except Exception:
            return 0.5  # Final fallback

    async def calculate_persona_complementarity_chatgpt(
        self, persona1: str, persona2: str
    ) -> float:
        """Calculate persona complementarity using ChatGPT"""
        
        prompt = f"""Rate the strategic complementarity between these two professional personas for networking/collaboration on a scale of 0.0 to 1.0.

Persona 1: {persona1}
Persona 2: {persona2}

Consider:
- Do they have complementary skills that would benefit collaboration?
- Would they bring different perspectives to solving problems?
- Are there natural synergies in their expertise domains?
- Would they learn valuable things from each other?

Rate from 0.0 (no complementarity/overlap) to 1.0 (perfect complementarity).

Respond with only a number between 0.0 and 1.0."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10,
            )
            
            result = response.choices[0].message.content.strip()
            score = float(result)
            return max(0.0, min(1.0, score))  # Clamp to [0,1]
            
        except Exception as e:
            print(f"ChatGPT error for {persona1} <-> {persona2}: {e}")
            # Fall back to embedding similarity
            return await self.calculate_embedding_similarity_fallback(persona1, persona2)

    def load_experience_matrix_from_redis(self) -> bool:
        """Load experience complementarity matrix from Redis cache"""
        cached_matrix = self.cache.get("experience_complementarity_matrix")
        if cached_matrix:
            try:
                self.experience_matrix = json.loads(cached_matrix)
                return True
            except json.JSONDecodeError:
                return False
        return False

    async def build_experience_complementarity_matrix(self, csv_path: str) -> Dict[str, Dict[str, float]]:
        """Build experience complementarity matrix using ChatGPT"""
        
        # Extract all unique experience tags from dataset
        import pandas as pd
        from tag_extractor import tag_extractor
        
        df = pd.read_csv(csv_path)
        all_experience_tags = set()
        
        for idx, row in df.iterrows():
            if pd.notna(row.get("Professional Identity - Experience Level", "")):
                tags = tag_extractor.extract_tags(row["Professional Identity - Experience Level"], "experience")
                all_experience_tags.update(tags)
        
        experience_list = list(all_experience_tags)
        print(f"Building experience complementarity matrix for {len(experience_list)} experience levels...")
        
        # Build matrix
        experience_matrix = {}
        for experience1 in experience_list:
            experience_matrix[experience1] = {}
            for experience2 in experience_list:
                if experience1 == experience2:
                    experience_matrix[experience1][experience2] = 0.0  # Same experience = no complementarity
                else:
                    # Use ChatGPT to score experience complementarity
                    try:
                        score = await self.calculate_experience_complementarity_chatgpt(experience1, experience2)
                        experience_matrix[experience1][experience2] = score
                    except Exception as e:
                        print(f"Error calculating {experience1} <-> {experience2}: {e}")
                        # Fallback based on semantic similarity
                        score = await self.calculate_embedding_similarity_fallback(experience1, experience2)
                        experience_matrix[experience1][experience2] = score

        # Cache the matrix
        self.cache.set("experience_complementarity_matrix", json.dumps(experience_matrix))
        self.experience_matrix = experience_matrix
        
        print(f"✅ Built experience complementarity matrix with {len(experience_list)} experience levels")
        return experience_matrix

    async def calculate_experience_complementarity_chatgpt(self, exp1: str, exp2: str) -> float:
        """Calculate experience complementarity using ChatGPT"""
        
        prompt = f"""Rate the complementarity value for professional networking between these two experience levels on a scale of 0.0 to 1.0.

Experience Level 1: {exp1}
Experience Level 2: {exp2}

Consider:
- Would they benefit from mentorship relationships (senior-junior)?
- Do they bring different perspectives and knowledge levels?
- Is there mutual learning potential?
- Would they complement each other in team dynamics?

Examples:
- Senior + Junior = High complementarity (0.7-0.9) - mentorship value
- Senior + Senior = Medium complementarity (0.3-0.5) - peer collaboration  
- Junior + Junior = Low complementarity (0.1-0.3) - similar skill levels

Rate from 0.0 (no complementarity) to 1.0 (perfect complementarity).
Respond with only a number between 0.0 and 1.0."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10,
            )
            
            result = response.choices[0].message.content.strip()
            score = float(result)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"ChatGPT error for {exp1} <-> {exp2}: {e}")
            return await self.calculate_embedding_similarity_fallback(exp1, exp2)

    async def calculate_experience_complementarity_fast(
        self, person1_experience: List[str], person2_experience: List[str]
    ) -> float:
        """Fast calculation of experience complementarity using cached matrix"""
        
        if not hasattr(self, "experience_matrix") or not self.experience_matrix:
            # Try to load from cache first
            if self.load_experience_matrix_from_redis():
                pass  # Successfully loaded
            else:
                # Use embedding similarity fallback
                if not person1_experience or not person2_experience:
                    return 0.0
                    
                # Calculate average embedding-based complementarity
                total_score = 0.0
                pair_count = 0
                
                for exp1 in person1_experience:
                    for exp2 in person2_experience:
                        score = await self.calculate_embedding_similarity_fallback(exp1, exp2)
                        total_score += score
                        pair_count += 1
                
                return total_score / pair_count if pair_count > 0 else 0.0

        if not person1_experience or not person2_experience:
            return 0.0

        total_score = 0.0
        pair_count = 0

        for exp1 in person1_experience:
            for exp2 in person2_experience:
                if (
                    exp1 in self.experience_matrix
                    and exp2 in self.experience_matrix[exp1]
                ):
                    total_score += self.experience_matrix[exp1][exp2]
                    pair_count += 1

        return total_score / pair_count if pair_count > 0 else 0.0


async def build_and_save_causal_graph(csv_path: str):
    """Build causal relationship graph and save to file"""
    analyzer = CausalRelationshipAnalyzer()

    # Build the graph
    causal_graph = await analyzer.build_causal_relationship_graph(csv_path)

    # Save to file
    analyzer.save_causal_graph()

    # Print summary
    analyzer.print_causal_graph_summary()

    return analyzer


# Usage example
if __name__ == "__main__":

    async def main():
        analyzer = await build_and_save_causal_graph("data/test_batch2.csv")

        # Example usage of the analyzer
        person1_tags = {
            "industry": ["SaaS", "B2B"],
            "market": ["Early-stage", "Series A"],
            "offering": ["API Platform"],
        }

        person2_tags = {
            "industry": ["Manufacturing", "Hardware"],
            "market": ["Growth-stage", "Enterprise"],
            "offering": ["Physical Products"],
        }

        score = analyzer.calculate_business_complementarity(person1_tags, person2_tags)
        print(f"\nBusiness complementarity score: {score:.3f}")

    asyncio.run(main())
