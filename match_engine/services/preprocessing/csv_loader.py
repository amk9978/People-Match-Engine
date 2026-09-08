import logging
from typing import Any, Dict, Optional

import pandas as pd

from match_engine.services.features.feature_set import FeatureSet
from match_engine.services.features.schema_mapper import (
    LoadPlan,
    SchemaError,
    SchemaMapper,
)

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
        mapping = self._read_mapping()
        plan = self.schema_mapper.load_plan(mapping or {})

        raw = pd.read_csv(self.csv_path, skiprows=plan.skip_rows)
        raw = self._join_name_columns(raw, plan)
        original_count = len(raw)

        self.feature_set = self._resolve_feature_set(raw, mapping)
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

    def _read_mapping(self) -> Optional[Dict[str, Any]]:
        if not self.mapping_path:
            return None
        return self.schema_mapper.read_mapping(self.mapping_path)

    def _resolve_feature_set(
        self, df: pd.DataFrame, mapping: Optional[Dict[str, Any]]
    ) -> FeatureSet:
        if mapping is not None:
            return self.schema_mapper.from_mapping(mapping, df)
        return self.schema_mapper.detect(df)

    def _join_name_columns(self, df: pd.DataFrame, plan: LoadPlan) -> pd.DataFrame:
        """Compose one display name from the columns the export split it across."""
        if not plan.name_columns:
            return df

        missing = [column for column in plan.name_columns if column not in df.columns]
        if missing:
            raise SchemaError(f"name_columns not in the dataset: {missing}")

        parts = [
            df[column].fillna("").astype(str).str.strip()
            for column in plan.name_columns
        ]
        joined = parts[0]
        for part in parts[1:]:
            joined = (joined + " " + part).str.strip()

        df = df.copy()
        df[plan.name_column] = joined.str.replace(r"\s+", " ", regex=True)
        return df

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
