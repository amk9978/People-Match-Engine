#!/usr/bin/env python3

import asyncio
import json
import os
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from embedding_service import embedding_service
from redis_cache import RedisEmbeddingCache
from tag_extractor import tag_extractor

load_dotenv()


class SemanticPersonDeduplicator:
    """Deduplicate tags within each person using cosine similarity clustering"""

    def __init__(self):
        self.cache = RedisEmbeddingCache()

    async def get_tag_embedding(self, tag: str) -> np.ndarray:
        """Get embedding using shared embedding service"""
        embedding = await embedding_service.get_embedding_array(tag)
        return embedding

    async def select_umbrella_tag(self, tag_cluster: List[str], category: str) -> str:
        if len(tag_cluster) == 1:
            return tag_cluster[0]

        cache_key = f"umbrella_selection:{category}:{sorted(tag_cluster)}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result

        tag_list = "\n".join([f"- {tag}" for tag in tag_cluster])

        prompt = f"""You are selecting the most canonical/representative term from these semantically similar {category} tags:

{tag_list}

Select the ONE tag that is:
- Most commonly used in professional contexts
- Clear and concise
- Industry-standard terminology

Return ONLY the selected tag, nothing else."""

        try:
            response = await embedding_service.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50,
            )

            selected_tag = response.choices[0].message.content.strip()

            if selected_tag in tag_cluster:
                self.cache.set(cache_key, selected_tag)
                return selected_tag
            else:
                return tag_cluster[0]

        except Exception as e:
            print(f"Error selecting umbrella tag: {e}")
            return tag_cluster[0]

    async def deduplicate_person_tags_semantic(
        self, tags: List[str], category: str, similarity_threshold: float = 0.80
    ) -> List[str]:
        if len(tags) <= 1:
            return tags

        embeddings = []
        valid_tags = []

        for tag in tags:
            embedding = await self.get_tag_embedding(tag)
            if np.any(embedding):
                embeddings.append(embedding)
                valid_tags.append(tag)

        if len(valid_tags) <= 1:
            return valid_tags

        embeddings = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - similarity_threshold,
            metric="precomputed",
            linkage="average",
        ).fit(distance_matrix)

        clusters = {}
        for i, cluster_id in enumerate(clustering.labels_):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(valid_tags[i])

        deduplicated_tags = []
        for cluster_tags in clusters.values():
            umbrella_tag = await self.select_umbrella_tag(cluster_tags, category)
            deduplicated_tags.append(umbrella_tag)

        return deduplicated_tags

    async def process_dataset_semantic(
        self, csv_path: str, similarity_threshold: float = 0.80
    ) -> Dict[str, Dict]:
        print(
            f"Starting semantic person-level deduplication (threshold: {similarity_threshold})..."
        )

        df = pd.read_csv(csv_path)

        feature_columns = {
            "role_spec": "Professional Identity - Role Specification",
            "experience": "Professional Identity - Experience Level",
            "industry": "Company Identity - Industry Classification",
            "market": "Company Market - Market Traction",
            "offering": "Company Offering - Value Proposition",
            "personas": "All Persona Titles",
        }

        results = {}

        for category, column_name in feature_columns.items():
            print(f"\nProcessing {category} tags...")

            original_tag_count = 0
            deduplicated_tag_count = 0
            person_reductions = []

            for idx, row in df.iterrows():
                if pd.notna(row[column_name]):
                    raw_tags = tag_extractor.extract_tags(row[column_name], category)

                    if len(raw_tags) > 1:
                        deduplicated_tags = await self.deduplicate_person_tags_semantic(
                            raw_tags, category, similarity_threshold
                        )
                    else:
                        deduplicated_tags = raw_tags

                    original_count = len(raw_tags)
                    deduplicated_count = len(deduplicated_tags)

                    original_tag_count += original_count
                    deduplicated_tag_count += deduplicated_count

                    if original_count != deduplicated_count:
                        person_reductions.append(
                            {
                                "person": row.get("Person Name", f"Person_{idx}"),
                                "original": raw_tags,
                                "deduplicated": deduplicated_tags,
                                "reduction": original_count - deduplicated_count,
                                "clusters_found": len(set(deduplicated_tags)),
                            }
                        )

                if idx % 5 == 0:
                    print(f"  Processed {idx + 1}/{len(df)} people...")

            reduction_count = original_tag_count - deduplicated_tag_count
            reduction_ratio = (
                reduction_count / original_tag_count if original_tag_count > 0 else 0
            )

            results[category] = {
                "original_tag_instances": original_tag_count,
                "deduplicated_tag_instances": deduplicated_tag_count,
                "total_reduction": reduction_count,
                "reduction_ratio": reduction_ratio,
                "people_affected": len(person_reductions),
                "similarity_threshold": similarity_threshold,
                "example_reductions": person_reductions[:5],
            }

            print(f"  Original tag instances: {original_tag_count}")
            print(f"  After deduplication: {deduplicated_tag_count}")
            print(f"  Total reduction: {reduction_count} ({reduction_ratio:.1%})")
            print(f"  People affected: {len(person_reductions)}")

        cache_key = f"semantic_person_dedup_{similarity_threshold}"
        self.cache.set(cache_key, json.dumps(results, default=str))

        return results

    async def apply_semantic_deduplication(
        self, tags: List[str], category: str, similarity_threshold: float = 0.80
    ) -> List[str]:
        """Apply semantic deduplication to a list of tags"""
        return await self.deduplicate_person_tags_semantic(
            tags, category, similarity_threshold
        )

    def get_stats(self, similarity_threshold: float = 0.80) -> Dict:
        """Get deduplication statistics"""
        cache_key = f"semantic_person_dedup_{similarity_threshold}"
        cached_results = self.cache.get(cache_key)
        if cached_results:
            try:
                return json.loads(cached_results)
            except json.JSONDecodeError:
                pass

        return {
            "error": f"No semantic deduplication results found for threshold {similarity_threshold}"
        }

    def print_example_reductions(self, similarity_threshold: float = 0.80):
        """Print examples of semantic tag reductions"""
        stats = self.get_stats(similarity_threshold)
        if "error" in stats:
            print("No semantic deduplication results available")
            return

        print("\n" + "=" * 60)
        print("SEMANTIC PERSON-LEVEL TAG DEDUPLICATION EXAMPLES")
        print(f"Similarity Threshold: {similarity_threshold}")
        print("=" * 60)

        for category, cat_stats in stats.items():
            if cat_stats.get("example_reductions"):
                print(f"\n📊 {category.upper()} Examples:")
                for example in cat_stats["example_reductions"]:
                    print(f"  {example['person']}:")
                    print(f"    Original: {example['original']}")
                    print(
                        f"    Semantic clusters → Umbrella tags: {example['deduplicated']}"
                    )
                    print(
                        f"    Reduction: {example['reduction']} tags → {example['clusters_found']} semantic groups"
                    )


# Usage example
async def run_semantic_deduplication(csv_path: str, similarity_threshold: float = 0.80):
    """Run semantic person-level deduplication"""
    deduplicator = SemanticPersonDeduplicator()

    # Process dataset
    results = await deduplicator.process_dataset_semantic(
        csv_path, similarity_threshold
    )

    # Print examples
    deduplicator.print_example_reductions(similarity_threshold)

    return deduplicator


if __name__ == "__main__":

    async def main():
        # Test with different thresholds
        for threshold in [0.75, 0.80, 0.85]:
            print(f"\n{'='*60}")
            print(f"TESTING THRESHOLD: {threshold}")
            print(f"{'='*60}")

            deduplicator = await run_semantic_deduplication(
                "data/batch2.csv", threshold
            )

            # Example usage
            test_tags = [
                "CTO",
                "Chief Technology Officer",
                "Software Developer",
                "AI Scientist",
            ]
            result = await deduplicator.apply_semantic_deduplication(
                test_tags, "role_spec", threshold
            )
            print(f"\nExample clustering: {test_tags} → {result}")

    asyncio.run(main())
