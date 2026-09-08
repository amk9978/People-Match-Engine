import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

from match_engine import settings
from match_engine.services.features.feature_set import WHOLE_CELL, Feature, FeatureSet

logger = logging.getLogger(__name__)

CANDIDATE_SEPARATORS = ("|", ";", ",")
NAME_HINTS = ("person name", "full name", "name", "attendee")
COMPANY_HINTS = ("company", "organisation", "organization", "employer")
SKIPPED_HINTS = ("url", "link", "email", "phone", "linkedin", "twitter", "id")
MIN_DISTINCT_VALUES = 2


class SchemaError(ValueError):
    """The dataset and the requested mapping cannot produce a usable feature set."""


class SchemaMapper:
    """Turns a CSV into a FeatureSet, from an explicit mapping or by inspection.

    Detection ranks text columns by how many distinct values they hold, because a
    column that repeats one value cannot separate two people.
    """

    def __init__(self, max_features: int = None):
        self.max_features = max_features or settings.MAX_FEATURES
        assert self.max_features > 0, "max_features must be positive"

    def from_file(self, mapping_path: str, df: pd.DataFrame) -> FeatureSet:
        mapping = yaml.safe_load(Path(mapping_path).read_text())
        if not isinstance(mapping, dict):
            raise SchemaError(f"{mapping_path} does not contain a mapping")
        return self.from_mapping(mapping, df)

    def from_mapping(self, mapping: Dict[str, Any], df: pd.DataFrame) -> FeatureSet:
        """Build a feature set from an explicit mapping, filling in any separator
        the mapping leaves unstated."""
        declared = mapping.get("features")
        if not declared:
            raise SchemaError("mapping declares no features")

        if len(declared) > self.max_features:
            raise SchemaError(
                f"mapping declares {len(declared)} features, the cap is "
                f"{self.max_features}; drop the least useful columns or raise "
                f"MAX_FEATURES"
            )

        features = []
        for entry in declared:
            column = entry["column"]
            if column not in df.columns:
                raise SchemaError(f"column {column} is not in the dataset")

            separator = entry.get("separator")
            if separator is None:
                separator = self._infer_separator(df[column])

            features.append(
                Feature(name=entry["name"], column=column, separator=separator)
            )

        name_column = mapping.get("name_column") or self._detect_name_column(df)
        company_column = mapping.get("company_column", "")
        if company_column and company_column not in df.columns:
            raise SchemaError(f"column {company_column} is not in the dataset")

        return FeatureSet(
            features=tuple(features),
            name_column=name_column,
            company_column=company_column,
        )

    def detect(self, df: pd.DataFrame) -> FeatureSet:
        """Infer a feature set from the dataset's own columns."""
        name_column = self._detect_name_column(df)
        company_column = self._detect_company_column(df, name_column)
        reserved = {name_column, company_column}

        ranked = self._rank_candidates(df, reserved)
        if not ranked:
            raise SchemaError(
                "no column carries enough distinct text to score; supply a "
                "mapping file naming the feature columns"
            )

        chosen = ranked[: self.max_features]
        if len(ranked) > self.max_features:
            dropped = [column for column, _ in ranked[self.max_features :]]
            logger.warning(
                f"Dataset has {len(ranked)} scoreable columns, capping at "
                f"{self.max_features}; ignoring {dropped}"
            )

        features = tuple(
            Feature(
                name=self._feature_name(column),
                column=column,
                separator=self._infer_separator(df[column]),
            )
            for column, _ in chosen
        )

        logger.info(
            f"Detected {len(features)} features: "
            f"{ {feature.name: feature.column for feature in features} }"
        )
        return FeatureSet(
            features=features,
            name_column=name_column,
            company_column=company_column,
        )

    def _rank_candidates(
        self, df: pd.DataFrame, reserved: set
    ) -> List[Tuple[str, int]]:
        candidates = []
        for column in df.columns:
            if column in reserved or self._is_skipped(column):
                continue
            values = df[column].dropna().astype(str).str.strip()
            values = values[values != ""]
            distinct = values.nunique()
            if distinct >= MIN_DISTINCT_VALUES:
                candidates.append((column, distinct))

        return sorted(candidates, key=lambda pair: (-pair[1], pair[0]))

    def _is_skipped(self, column: str) -> bool:
        lowered = column.lower()
        return any(hint in lowered for hint in SKIPPED_HINTS)

    def _detect_name_column(self, df: pd.DataFrame) -> str:
        for hint in NAME_HINTS:
            for column in df.columns:
                if hint in column.lower():
                    return column
        raise SchemaError(
            "no column names each person; add name_column to a mapping file"
        )

    def _detect_company_column(self, df: pd.DataFrame, name_column: str) -> str:
        for hint in COMPANY_HINTS:
            for column in df.columns:
                if column != name_column and hint in column.lower():
                    return column
        return ""

    def _feature_name(self, column: str) -> str:
        """Reduce a column header to a short lowercase feature name."""
        tail = column.split("-")[-1].strip()
        words = [word for word in tail.replace("_", " ").split() if word.isalpha()]
        if not words:
            words = [
                word for word in column.replace("_", " ").split() if word.isalpha()
            ]
        return "_".join(word.lower() for word in words[-2:])

    def _infer_separator(self, values: pd.Series) -> str:
        """Pick the separator that appears in the most cells, if any does."""
        text = values.dropna().astype(str)
        if text.empty:
            return WHOLE_CELL

        counts = {
            separator: int(text.str.contains(separator, regex=False).sum())
            for separator in CANDIDATE_SEPARATORS
        }
        best = max(counts, key=counts.get)
        if counts[best] < len(text) / 2:
            return WHOLE_CELL
        return best
