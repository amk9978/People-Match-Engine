from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

WHOLE_CELL = ""


@dataclass(frozen=True)
class Feature:
    """One scored dimension: a name, the column carrying it, and how to split it.

    An empty separator means the whole cell is a single tag.
    """

    name: str
    column: str
    separator: str = WHOLE_CELL

    def __post_init__(self):
        assert self.name, "a feature needs a name"
        assert self.column, f"feature {self.name} needs a column"

    def split(self, cell: str) -> List[str]:
        if self.separator == WHOLE_CELL:
            tag = cell.strip()
            if tag:
                return [tag]
            return []
        return [tag.strip() for tag in cell.split(self.separator) if tag.strip()]


@dataclass(frozen=True)
class FeatureSet:
    """The features one analysis run scores, derived from the dataset at runtime."""

    features: Tuple[Feature, ...]
    name_column: str
    company_column: str = ""

    def __post_init__(self):
        assert self.features, "an analysis needs at least one feature"
        assert self.name_column, "an analysis needs a column naming each person"

        names = [feature.name for feature in self.features]
        assert len(names) == len(set(names)), f"feature names must be unique: {names}"

    def __iter__(self) -> Iterator[Feature]:
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    @property
    def columns(self) -> Dict[str, str]:
        return {feature.name: feature.column for feature in self.features}

    def feature(self, name: str) -> Feature:
        for feature in self.features:
            if feature.name == name:
                return feature
        raise KeyError(f"no feature named {name}")

    def required_columns(self) -> Tuple[str, ...]:
        columns = [self.name_column]
        if self.company_column:
            columns.append(self.company_column)
        columns.extend(feature.column for feature in self.features)
        return tuple(dict.fromkeys(columns))
