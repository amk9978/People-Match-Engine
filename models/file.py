#!/usr/bin/env python3

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class FileVersion:
    """File version model for tracking dataset modifications"""

    version_id: str
    file_id: str
    version_number: int
    created_at: datetime
    description: str
    changes_summary: Dict[str, Any]
    data_hash: str
    row_count: int
    column_count: int
    file_size: int
    file_path: Optional[str] = None  # Path to actual file data in /data

    def __post_init__(self):
        if not self.version_id:
            self.version_id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "version_id": self.version_id,
            "file_id": self.file_id,
            "version_number": self.version_number,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "changes_summary": self.changes_summary,
            "data_hash": self.data_hash,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "file_size": self.file_size,
            "file_path": self.file_path,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FileVersion":
        return cls(
            version_id=data["version_id"],
            file_id=data["file_id"],
            version_number=data["version_number"],
            created_at=datetime.fromisoformat(data["created_at"]),
            description=data["description"],
            changes_summary=data["changes_summary"],
            data_hash=data["data_hash"],
            row_count=data["row_count"],
            column_count=data["column_count"],
            file_size=data["file_size"],
            file_path=data.get("file_path"),
        )


@dataclass
class File:
    """File/Dataset model with versioning support"""

    file_id: str
    user_id: str
    filename: str
    original_filename: str
    created_at: datetime
    updated_at: datetime
    current_version_id: str
    total_versions: int = 1
    total_jobs: int = 0
    file_type: str = "csv"
    metadata: Dict[str, Any] = None
    tags: List[str] = None

    def __post_init__(self):
        if not self.file_id:
            self.file_id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict:
        return {
            "file_id": self.file_id,
            "user_id": self.user_id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "current_version_id": self.current_version_id,
            "total_versions": self.total_versions,
            "total_jobs": self.total_jobs,
            "file_type": self.file_type,
            "metadata": self.metadata,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "File":
        return cls(
            file_id=data["file_id"],
            user_id=data["user_id"],
            filename=data["filename"],
            original_filename=data["original_filename"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            current_version_id=data["current_version_id"],
            total_versions=data.get("total_versions", 1),
            total_jobs=data.get("total_jobs", 0),
            file_type=data.get("file_type", "csv"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )

    def update_timestamp(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.now()

    def increment_version_count(self):
        """Increment total versions and update timestamp"""
        self.total_versions += 1
        self.update_timestamp()

    def increment_job_count(self):
        """Increment total jobs count"""
        self.total_jobs += 1


@dataclass
class FileStats:
    """File statistics model"""

    file_id: str
    filename: str
    user_id: str
    total_versions: int
    total_jobs: int
    current_size: int
    created_at: datetime
    last_modified: datetime
    last_job_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "user_id": self.user_id,
            "total_versions": self.total_versions,
            "total_jobs": self.total_jobs,
            "current_size": self.current_size,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "last_job_at": self.last_job_at.isoformat() if self.last_job_at else None,
        }


@dataclass
class DatasetDiff:
    """Dataset difference model for comparing versions"""

    version1_id: str
    version2_id: str
    added_rows: List[int]
    deleted_rows: List[int]
    modified_rows: List[int]
    added_columns: List[str]
    deleted_columns: List[str]
    summary: Dict[str, int]

    def to_dict(self) -> Dict:
        return {
            "version1_id": self.version1_id,
            "version2_id": self.version2_id,
            "added_rows": self.added_rows,
            "deleted_rows": self.deleted_rows,
            "modified_rows": self.modified_rows,
            "added_columns": self.added_columns,
            "deleted_columns": self.deleted_columns,
            "summary": self.summary,
        }
