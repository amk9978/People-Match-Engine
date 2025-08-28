
import asyncio
import json
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from services.embedding_service import embedding_service
from services.redis_cache import RedisEmbeddingCache
from services.tag_extractor import tag_extractor

load_dotenv()

logger = logging.getLogger(__name__)


class SemanticPersonDeduplicator:
    """Deduplicate tags within each person using cosine similarity clustering"""

    def __init__(self):
        self.cache = RedisEmbeddingCache()
        self.embedding_service = embedding_service

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
                temperature=0,
                max_tokens=50,
            )

            selected_tag = response.choices[0].message.content.strip()

            if selected_tag in tag_cluster:
                self.cache.set(cache_key, selected_tag)
                return selected_tag
            else:
                return tag_cluster[0]

        except Exception as e:
            logger.info(f"Error selecting umbrella tag: {e}")
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
        logger.info(
            f"Starting dataset-wide semantic deduplication (threshold: {similarity_threshold})..."
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
            logger.info(f"\n🔍 Processing {category} tags...")

            # STEP 1: Extract all unique tags from entire dataset
            all_unique_tags = set()
            for idx, row in df.iterrows():
                if pd.notna(row[column_name]):
                    raw_tags = tag_extractor.extract_tags(row[column_name], category)
                    all_unique_tags.update(raw_tags)

            unique_tags_list = list(all_unique_tags)
            logger.info(f"  Found {len(unique_tags_list)} unique tags across dataset")

            # STEP 2: Create global tag mapping (tag -> umbrella_tag)
            tag_mapping = await self.create_global_tag_mapping(
                unique_tags_list, category, similarity_threshold
            )

            mapped_count = len([k for k, v in tag_mapping.items() if k != v])
            logger.info(f"  Created mappings for {mapped_count} tags")

            # Debug: Show sample mappings
            sample_mappings = {k: v for k, v in tag_mapping.items() if k != v}
            if sample_mappings:
                logger.info("  Sample mappings:")
                for orig, mapped in list(sample_mappings.items())[:3]:
                    logger.info(f"    '{orig}' → '{mapped}'")
            else:
                logger.info(
                    "  🚨 WARNING: No mappings created - all tags considered unique!"
                )

            # STEP 3: Apply mappings to dataset and track statistics
            original_tag_count = 0
            deduplicated_tag_count = 0
            person_reductions = []
            total_mappings_applied = 0

            for idx, row in df.iterrows():
                if pd.notna(row[column_name]):
                    raw_tags = tag_extractor.extract_tags(row[column_name], category)

                    # Apply global mapping and track what changed
                    mapped_tags = [tag_mapping.get(tag, tag) for tag in raw_tags]
                    deduplicated_tags = list(
                        set(mapped_tags)
                    )  # Remove duplicates after mapping

                    # Count how many tags were actually mapped
                    mappings_applied = sum(
                        1 for i, tag in enumerate(raw_tags) if mapped_tags[i] != tag
                    )
                    total_mappings_applied += mappings_applied

                    original_count = len(raw_tags)
                    deduplicated_count = len(deduplicated_tags)

                    original_tag_count += original_count
                    deduplicated_tag_count += deduplicated_count

                    if original_count != deduplicated_count or mappings_applied > 0:
                        person_reductions.append(
                            {
                                "person": row.get("Person Name", f"Person_{idx}"),
                                "original": raw_tags,
                                "mapped": mapped_tags,
                                "deduplicated": deduplicated_tags,
                                "reduction": original_count - deduplicated_count,
                                "mappings_applied": mappings_applied,
                            }
                        )

            logger.info(f"  🔄 Total mappings applied: {total_mappings_applied}")

            reduction_count = original_tag_count - deduplicated_tag_count
            reduction_ratio = (
                reduction_count / original_tag_count if original_tag_count > 0 else 0
            )

            results[category] = {
                "unique_tags_found": len(unique_tags_list),
                "tag_mappings_created": len(
                    [k for k, v in tag_mapping.items() if k != v]
                ),
                "original_tag_instances": original_tag_count,
                "deduplicated_tag_instances": deduplicated_tag_count,
                "total_reduction": reduction_count,
                "reduction_ratio": reduction_ratio,
                "people_affected": len(person_reductions),
                "similarity_threshold": similarity_threshold,
                "example_reductions": person_reductions[:5],
                "sample_mappings": dict(
                    list({k: v for k, v in tag_mapping.items() if k != v}.items())[:10]
                ),
            }

            logger.info(
                f"  ✅ Tag instances: {original_tag_count} → {deduplicated_tag_count}"
            )
            logger.info(
                f"  📉 Total reduction: {reduction_count} ({reduction_ratio:.1%})"
            )
            logger.info(f"  👥 People affected: {len(person_reductions)}")

        cache_key = f"semantic_dataset_dedup_{similarity_threshold}"
        self.cache.set(cache_key, json.dumps(results, default=str))

        return results

    async def create_global_tag_mapping(
        self, unique_tags: List[str], category: str, similarity_threshold: float
    ) -> Dict[str, str]:
        """Create global tag->umbrella mapping for consistent deduplication"""

        # TODO:
        # if len(unique_tags) <= 1:
        return {tag: tag for tag in unique_tags}

        # Get embeddings for all unique tags
        embeddings = await self.embedding_service.get_batch_embeddings(unique_tags)

        # Calculate similarity matrix
        similarities = cosine_similarity(embeddings)

        # Find clusters of similar tags
        tag_mapping = {}
        processed_tags = set()

        for i, tag1 in enumerate(unique_tags):
            if tag1 in processed_tags:
                continue

            # Find all tags similar to this one
            similar_tags = [tag1]  # Include the tag itself

            for j, tag2 in enumerate(unique_tags):
                if i != j and tag2 not in processed_tags:
                    if similarities[i][j] >= similarity_threshold:
                        similar_tags.append(tag2)

            if len(similar_tags) > 1:
                # Use ChatGPT to select umbrella term for this cluster
                umbrella_tag = await self.select_umbrella_tag(similar_tags, category)

                # Map all similar tags to umbrella
                for tag in similar_tags:
                    tag_mapping[tag] = umbrella_tag
                    processed_tags.add(tag)
            else:
                # Single tag, maps to itself
                tag_mapping[tag1] = tag1
                processed_tags.add(tag1)

        return tag_mapping

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
            logger.info("No semantic deduplication results available")
            return

        logger.info("\n" + "=" * 60)
        logger.info("SEMANTIC PERSON-LEVEL TAG DEDUPLICATION EXAMPLES")
        logger.info(f"Similarity Threshold: {similarity_threshold}")
        logger.info("=" * 60)

        for category, cat_stats in stats.items():
            if cat_stats.get("example_reductions"):
                logger.info(f"\n📊 {category.upper()} Examples:")
                for example in cat_stats["example_reductions"]:
                    logger.info(f"  {example['person']}:")
                    logger.info(f"    Original: {example['original']}")
                    logger.info(
                        f"    Semantic clusters → Umbrella tags: {example['deduplicated']}"
                    )
                    logger.info(
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
            logger.info(f"\n{'=' * 60}")
            logger.info(f"TESTING THRESHOLD: {threshold}")
            logger.info(f"{'=' * 60}")

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
            logger.info(f"\nExample clustering: {test_tags} → {result}")

    asyncio.run(main())
