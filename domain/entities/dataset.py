#!/usr/bin/env python3

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

import pandas as pd


@dataclass
class DatasetVersion:
    """Dataset version domain entity"""

    version_id: str
    operation_type: str
    created_at: datetime
    row_count: int
    changes: Dict[str, int]
    file_path: str
    description: str

    @classmethod
    def create_original(cls, row_count: int, file_path: str) -> "DatasetVersion":
        """Create original version"""
        return cls(
            version_id=str(uuid4())[:8],
            operation_type="original",
            created_at=datetime.utcnow(),
            row_count=row_count,
            changes={"added_rows": 0, "deleted_rows": 0},
            file_path=file_path,
            description="Original uploaded dataset",
        )

    @classmethod
    def create_modification(
        cls,
        operation_type: str,
        row_count: int,
        changes: Dict[str, int],
        file_path: str,
        description: str,
    ) -> "DatasetVersion":
        """Create modification version"""
        return cls(
            version_id=str(uuid4())[:8],
            operation_type=operation_type,
            created_at=datetime.utcnow(),
            row_count=row_count,
            changes=changes,
            file_path=file_path,
            description=description,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            "version_id": self.version_id,
            "type": self.operation_type,
            "created_at": self.created_at.isoformat(),
            "row_count": self.row_count,
            "changes": self.changes,
            "file_path": self.file_path,
            "description": self.description,
        }


@dataclass
class Dataset:
    """Dataset domain entity"""

    user_id: str
    filename: str
    original_version_id: str
    current_version_id: str
    created_at: datetime
    column_count: int
    columns: List[str]
    versions: List[DatasetVersion]

    @property
    def current_row_count(self) -> int:
        """Get current row count from latest version"""
        current_version = self.get_version(self.current_version_id)
        return current_version.row_count if current_version else 0

    @property
    def total_versions(self) -> int:
        """Get total number of versions"""
        return len(self.versions)

    def get_version(self, version_id: str) -> Optional[DatasetVersion]:
        """Get specific version by ID"""
        return next((v for v in self.versions if v.version_id == version_id), None)

    def add_version(self, version: DatasetVersion) -> None:
        """Add new version"""
        self.versions.append(version)
        self.current_version_id = version.version_id

    def get_version_diff(self, version1_id: str, version2_id: str) -> Dict:
        """Get differences between two versions"""
        v1 = self.get_version(version1_id)
        v2 = self.get_version(version2_id)

        if not v1 or not v2:
            raise ValueError("One or both versions not found")

        return {
            "version1": {"id": version1_id, "rows": v1.row_count},
            "version2": {"id": version2_id, "rows": v2.row_count},
            "row_difference": v2.row_count - v1.row_count,
            "columns_changed": [],  # Would need actual comparison
            "summary": f"Version {version2_id} has {v2.row_count - v1.row_count:+d} rows compared to {version1_id}",
        }

    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            "user_id": self.user_id,
            "filename": self.filename,
            "original_version_id": self.original_version_id,
            "current_version_id": self.current_version_id,
            "created_at": self.created_at.isoformat(),
            "row_count": self.current_row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "versions": [v.to_dict() for v in self.versions],
            "total_versions": self.total_versions,
        }


@dataclass
class DatasetModification:
    """Value object for dataset modifications"""

    rows: List[Dict]
    description: Optional[str] = None

    def validate_columns(self, expected_columns: List[str]) -> None:
        """Validate that all rows have expected columns"""
        if not self.rows:
            return

        for i, row in enumerate(self.rows):
            missing_cols = set(expected_columns) - set(row.keys())
            if missing_cols:
                raise ValueError(f"Row {i} missing columns: {missing_cols}")

            extra_cols = set(row.keys()) - set(expected_columns)
            if extra_cols:
                raise ValueError(f"Row {i} has extra columns: {extra_cols}")


@dataclass
class DatasetDeletionCriteria:
    """Value object for dataset deletion criteria"""

    criteria: Dict
    description: Optional[str] = None

    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply deletion criteria to dataframe"""
        mask = pd.Series([True] * len(df), index=df.index)

        for column, value in self.criteria.items():
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in dataset")

            if isinstance(value, dict):
                operator = value.get("operator", "equals")
                search_value = value.get("value")

                if operator == "equals":
                    mask = mask & (df[column] == search_value)
                elif operator == "contains":
                    mask = mask & df[column].astype(str).str.contains(
                        str(search_value), na=False
                    )
                elif operator == "not_equals":
                    mask = mask & (df[column] != search_value)
                elif operator == "empty":
                    mask = mask & (
                        df[column].isna() | (df[column].astype(str).str.strip() == "")
                    )
                elif operator == "not_empty":
                    mask = mask & (
                        df[column].notna() & (df[column].astype(str).str.strip() != "")
                    )
            else:
                mask = mask & (df[column] == value)

        return df[mask]
