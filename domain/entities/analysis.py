#!/usr/bin/env python3

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from uuid import uuid4


class JobStatus(Enum):
    """Analysis job status enumeration"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AnalysisJob:
    """Analysis job domain entity"""
    job_id: str
    user_id: str
    filename: str
    status: JobStatus
    created_at: datetime
    min_density: Optional[float] = None
    prompt: Optional[str] = None
    version_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    @classmethod
    def create(cls, user_id: str, filename: str, min_density: Optional[float] = None,
               prompt: Optional[str] = None, version_id: Optional[str] = None) -> 'AnalysisJob':
        """Create new analysis job"""
        return cls(
            job_id=str(uuid4()),
            user_id=user_id,
            filename=filename,
            status=JobStatus.QUEUED,
            created_at=datetime.utcnow(),
            min_density=min_density,
            prompt=prompt,
            version_id=version_id
        )
    
    def start(self) -> None:
        """Mark job as started"""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def complete(self) -> None:
        """Mark job as completed"""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
    
    def fail(self, error_message: str) -> None:
        """Mark job as failed"""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
    
    def cancel(self) -> None:
        """Cancel the job"""
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.utcnow()
    
    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds"""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "filename": self.filename,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "min_density": self.min_density,
            "prompt": self.prompt,
            "version_id": self.version_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": self.duration,
            "error_message": self.error_message
        }


@dataclass
class AnalysisResult:
    """Analysis result domain entity"""
    job_id: str
    user_id: str
    filename: str
    version_id: Optional[str]
    created_at: datetime
    subgraph_size: int
    subgraph_density: float
    total_edges: int
    insights: Dict[str, Any]
    execution_time: float
    hyperparameters: Dict[str, float]
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "filename": self.filename,
            "version_id": self.version_id,
            "created_at": self.created_at.isoformat(),
            "subgraph_size": self.subgraph_size,
            "subgraph_density": self.subgraph_density,
            "total_edges": self.total_edges,
            "insights": self.insights,
            "execution_time": self.execution_time,
            "hyperparameters": self.hyperparameters
        }