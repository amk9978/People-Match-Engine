#!/usr/bin/env python3

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class User:
    """User model"""

    user_id: str
    created_at: datetime
    last_active: datetime
    total_files: int = 0
    total_analyses: int = 0

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "total_files": self.total_files,
            "total_analyses": self.total_analyses,
        }


@dataclass
class UserFile:
    """User file model"""

    filename: str
    uploaded_at: datetime
    file_size: int
    analysis_count: int = 0
    last_analysis: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "filename": self.filename,
            "uploaded_at": self.uploaded_at.isoformat(),
            "file_size": self.file_size,
            "analysis_count": self.analysis_count,
            "last_analysis": (
                self.last_analysis.isoformat() if self.last_analysis else None
            ),
        }
