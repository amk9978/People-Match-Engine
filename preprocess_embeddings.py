#!/usr/bin/env python3

import asyncio
from typing import List, Set

import pandas as pd

from embedding_service import embedding_service
from tag_extractor import tag_extractor


class EmbeddingPreprocessor:
    """Preprocessing service to embed all unique tags in the dataset"""

    def __init__(self):
        self.embedding_service = embedding_service

    def extract_all_unique_tags(self, csv_path: str) -> Set[str]:
        """Extract all unique tags from the dataset"""
        df = pd.read_csv(csv_path)
        all_tags = set()

        # Column mappings
        columns = {
            "role_spec": "Professional Identity - Role Specification",
            "experience": "Professional Identity - Experience Level",
            "industry": "Company Identity - Industry Classification",
            "market": "Company Market - Market Traction",
            "offering": "Company Offering - Value Proposition",
            "personas": "All Persona Titles",
        }

        for tag_type, column_name in columns.items():
            if column_name in df.columns:
                print(f"Extracting {tag_type} tags from '{column_name}'...")

                for value in df[column_name].dropna():
                    # Use the unified extract_tags method
                    tags = tag_extractor.extract_tags(str(value), tag_type)

                    all_tags.update(tags)

        print(f"Total unique tags extracted: {len(all_tags)}")
        return all_tags

    async def preprocess_all_embeddings(self, csv_path: str) -> None:
        """Extract all unique curated tags and batch embed them"""
        print("🚀 Starting embedding preprocessing for curated tags...")

        # Extract all unique tags (now curated after semantic deduplication)
        all_tags = self.extract_all_unique_tags(csv_path)
        tags_list = list(all_tags)

        print(
            f"📋 Preprocessing curated tags (post-deduplication): {len(tags_list)} unique tags"
        )

        # Check which tags are already cached
        uncached_tags = []
        cached_count = 0

        for tag in tags_list:
            if self.embedding_service.cache.get(tag) is None:
                uncached_tags.append(tag)
            else:
                cached_count += 1

        print(
            f"📊 Cache status: {cached_count} already cached, {len(uncached_tags)} need embedding"
        )

        if not uncached_tags:
            print("✅ All tags already cached!")
            return

        # Batch embed all uncached tags
        print(f"🔄 Batch embedding {len(uncached_tags)} unique tags...")
        embeddings = await self.embedding_service.get_batch_embeddings(uncached_tags)

        # Validate results
        successful_embeddings = sum(1 for emb in embeddings if emb != [0.0] * 1536)
        failed_embeddings = len(embeddings) - successful_embeddings

        print(f"✅ Preprocessing complete!")
        print(f"   - Successfully embedded: {successful_embeddings}")
        print(f"   - Failed (using fallback): {failed_embeddings}")
        print(f"   - Total cache size: {cached_count + successful_embeddings}")

    async def verify_embeddings(self, csv_path: str) -> None:
        """Verify all tags in dataset have embeddings cached"""
        print("🔍 Verifying embedding coverage...")

        all_tags = self.extract_all_unique_tags(csv_path)
        missing_tags = []

        for tag in all_tags:
            if self.embedding_service.cache.get(tag) is None:
                missing_tags.append(tag)

        if missing_tags:
            print(f"❌ Missing embeddings for {len(missing_tags)} tags:")
            for tag in missing_tags[:10]:  # Show first 10
                print(f"   - '{tag}'")
            if len(missing_tags) > 10:
                print(f"   ... and {len(missing_tags) - 10} more")
        else:
            print("✅ All tags have cached embeddings!")


async def preprocess_dataset(csv_path: str):
    """Main preprocessing function"""
    preprocessor = EmbeddingPreprocessor()

    # Run preprocessing
    await preprocessor.preprocess_all_embeddings(csv_path)

    # Verify everything is cached
    await preprocessor.verify_embeddings(csv_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python preprocess_embeddings.py <csv_path>")
        print("Example: python preprocess_embeddings.py data/batch2.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    print(f"Preprocessing embeddings for: {csv_path}")

    asyncio.run(preprocess_dataset(csv_path))
