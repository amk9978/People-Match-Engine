from typing import List

import pandas as pd


class TagExtractor:
    """Utility for consistent tag parsing across all features"""

    @staticmethod
    def extract_tags(text: str, category: str) -> List[str]:
        """Extract tags based on category-specific separators"""
        if pd.isna(text):
            return []

        if category == "personas":

            raw_tags = [tag.strip() for tag in str(text).split(";")]
        else:

            raw_tags = [tag.strip() for tag in str(text).split("|")]

        return [tag for tag in raw_tags if tag]

    @staticmethod
    def extract_persona_tags(persona_titles: str) -> List[str]:
        """Extract persona tags (semicolon separated) - backwards compatibility"""
        return TagExtractor.extract_tags(persona_titles, "personas")


tag_extractor = TagExtractor()
