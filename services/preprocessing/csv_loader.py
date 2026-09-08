import logging
from typing import Optional

import pandas as pd

from services.features.feature_set import FeatureSet
from services.features.schema_mapper import SchemaMapper

logger = logging.getLogger(__name__)


class CSVLoader:
    """Loads a roster CSV, derives its feature set, and drops unusable rows.

    The feature set comes from a mapping file when one is given and from the
    dataset's own columns otherwise, so no column name is compiled into the
    engine. The returned frame is always indexed 0..n-1, because downstream code
    addresses people by position and a gapped index would score one person under
    another's name.
    """

    def __init__(
        self,
        csv_path: str,
        mapping_path: Optional[str] = None,
        schema_mapper: SchemaMapper = None,
    ):
        self.csv_path = csv_path
        self.mapping_path = mapping_path
        self.schema_mapper = schema_mapper or SchemaMapper()
        self.df = None
        self.feature_set: Optional[FeatureSet] = None

    def load_data(self) -> pd.DataFrame:
        raw = pd.read_csv(self.csv_path)
        original_count = len(raw)

        self.feature_set = self._resolve_feature_set(raw)
        self.df = self.filter_incomplete_rows(raw, self.feature_set)
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

    def _resolve_feature_set(self, df: pd.DataFrame) -> FeatureSet:
        if self.mapping_path:
            return self.schema_mapper.from_file(self.mapping_path, df)
        return self.schema_mapper.detect(df)

    def filter_incomplete_rows(
        self, df: pd.DataFrame, feature_set: FeatureSet
    ) -> pd.DataFrame:
        """Drop rows with a blank required column and reindex by position."""
        mask = pd.Series(True, index=df.index)

        for column in feature_set.required_columns():
            values = df[column].astype(str).str.strip()
            column_mask = df[column].notna() & (values != "") & (values != "nan")
            mask = mask & column_mask

            dropped = int((~column_mask).sum())
            if dropped:
                logger.info(f"Column {column}: {dropped} rows blank")

        return df[mask].reset_index(drop=True)
