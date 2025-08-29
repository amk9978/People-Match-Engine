import logging
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class CSVLoader:
    """Handles CSV data loading and preprocessing with row filtering"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the dataset with row filtering"""
        self.df = pd.read_csv(self.csv_path)
        original_count = len(self.df)
        logger.info(f"Loaded {original_count} people from dataset")

        # Filter out rows with empty essential columns
        self.df = self.filter_incomplete_rows(self.df)
        filtered_count = len(self.df)

        if filtered_count < original_count:
            removed_count = original_count - filtered_count
            logger.info(
                f"🗑️ Filtered out {removed_count} rows with missing essential data"
            )
            logger.info(f"📊 Using {filtered_count} complete rows for analysis")

        return self.df

    def filter_incomplete_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out rows with empty essential columns"""

        essential_columns = [
            "Person Name",
            "Person Title",
            "Person Company",
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
            logger.info(f"⚠️ Warning: Missing columns in dataset: {missing_columns}")

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
                logger.info(
                    f"   • {column}: {filtered_by_column} rows have empty/missing values"
                )

        # Apply the filter
        filtered_df = df[mask].copy()

        return filtered_df

    def get_feature_columns(self) -> dict:
        """Get the mapping of feature names to column names"""
        return {
            "role": "Professional Identity - Role Specification",
            "experience": "Professional Identity - Experience Level",
            "industry": "Company Identity - Industry Classification",
            "market": "Company Market - Market Traction",
            "offering": "Company Offering - Value Proposition",
            "personas": "All Persona Titles",
        }
