#!/usr/bin/env python3

from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
from uuid import UUID, uuid4


@dataclass
class User:
    """User domain entity"""
    user_id: str
    created_at: datetime
    last_active: datetime
    total_files: int = 0
    total_analyses: int = 0
    
    def update_activity(self) -> None:
        """Update user's last activity timestamp"""
        self.last_active = datetime.utcnow()
    
    def increment_file_count(self) -> None:
        """Increment total file count"""
        self.total_files += 1
    
    def decrement_file_count(self) -> None:
        """Decrement total file count"""
        if self.total_files > 0:
            self.total_files -= 1
    
    def increment_analysis_count(self) -> None:
        """Increment total analysis count"""
        self.total_analyses += 1


@dataclass
class UserFile:
    """User file domain entity"""
    filename: str
    uploaded_at: datetime
    file_size: int
    analysis_count: int = 0
    last_analysis: Optional[datetime] = None
    
    def increment_analysis_count(self) -> None:
        """Increment analysis count for this file"""
        self.analysis_count += 1
        self.last_analysis = datetime.utcnow()
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            "filename": self.filename,
            "uploaded_at": self.uploaded_at.isoformat(),
            "file_size": self.file_size,
            "analysis_count": self.analysis_count,
            "last_analysis": self.last_analysis.isoformat() if self.last_analysis else None
        }