import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

ESSENTIAL_COLUMNS = [
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


class CSVLoader:
    """Loads a roster CSV and drops rows missing data the analysis requires.

    The returned frame is always indexed 0..n-1. Downstream code addresses people
    by position, so a gapped index would score one person under another's name.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.csv_path)
        original_count = len(self.df)

        self.df = self.filter_incomplete_rows(self.df)
        removed_count = original_count - len(self.df)

        if removed_count:
            logger.info(
                f"Loaded {len(self.df)} of {original_count} people, "
                f"{removed_count} dropped for missing essential data"
            )
        else:
            logger.info(f"Loaded {original_count} people")

        assert self.df.index.equals(
            pd.RangeIndex(len(self.df))
        ), "loaded frame must be indexed by position"
        return self.df

    def filter_incomplete_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with a blank essential column and reindex by position."""
        present_columns = [
            column for column in ESSENTIAL_COLUMNS if column in df.columns
        ]
        missing_columns = [
            column for column in ESSENTIAL_COLUMNS if column not in df.columns
        ]

        if missing_columns:
            logger.warning(f"Dataset is missing expected columns: {missing_columns}")

        mask = pd.Series(True, index=df.index)

        for column in present_columns:
            values = df[column].astype(str).str.strip()
            column_mask = df[column].notna() & (values != "") & (values != "nan")
            mask = mask & column_mask

            dropped_by_column = int((~column_mask).sum())
            if dropped_by_column:
                logger.info(f"Column {column}: {dropped_by_column} rows blank")

        return df[mask].reset_index(drop=True)

    def get_feature_columns(self) -> dict:
        """Map feature names to the columns that carry them."""
        return {
            "role": "Professional Identity - Role Specification",
            "experience": "Professional Identity - Experience Level",
            "industry": "Company Identity - Industry Classification",
            "market": "Company Market - Market Traction",
            "offering": "Company Offering - Value Proposition",
            "personas": "All Persona Titles",
        }
