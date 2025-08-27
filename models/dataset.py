#!/usr/bin/env python3

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DatasetVersion:
    """Dataset version model"""
    version_id: str
    operation_type: str
    created_at: datetime
    row_count: int
    changes: Dict[str, int]
    file_path: str
    description: str
    
    def to_dict(self) -> Dict:
        return {
            "version_id": self.version_id,
            "type": self.operation_type,
            "created_at": self.created_at.isoformat(),
            "row_count": self.row_count,
            "changes": self.changes,
            "file_path": self.file_path,
            "description": self.description
        }


@dataclass
class Dataset:
    """Dataset model"""
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
        current_version = self.get_version(self.current_version_id)
        return current_version.row_count if current_version else 0
    
    @property
    def total_versions(self) -> int:
        return len(self.versions)
    
    def get_version(self, version_id: str) -> Optional[DatasetVersion]:
        return next((v for v in self.versions if v.version_id == version_id), None)
    
    def to_dict(self) -> Dict:
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
            "total_versions": self.total_versions
        }