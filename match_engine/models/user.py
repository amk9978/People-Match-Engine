#!/usr/bin/env python3

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


@dataclass
class User:
    id: str
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class UserStats:
    """User statistics - the only user data we need to track"""

    user_id: str
    total_files: int = 0
    total_jobs: int = 0
    total_analyses: int = 0
    storage_used: int = 0
    last_activity: Optional[datetime] = None

    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "total_files": self.total_files,
            "total_jobs": self.total_jobs,
            "total_analyses": self.total_analyses,
            "storage_used": self.storage_used,
            "last_activity": (
                self.last_activity.isoformat() if self.last_activity else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UserStats":
        return cls(
            user_id=data["user_id"],
            total_files=data.get("total_files", 0),
            total_jobs=data.get("total_jobs", 0),
            total_analyses=data.get("total_analyses", 0),
            storage_used=data.get("storage_used", 0),
            last_activity=(
                datetime.fromisoformat(data["last_activity"])
                if data.get("last_activity")
                else None
            ),
        )

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
