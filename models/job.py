#!/usr/bin/env python3

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    """Job status enumeration"""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Job type enumeration"""

    ANALYSIS = "analysis"
    GRAPH_MATCHING = "graph_matching"
    SUBGRAPH_ANALYSIS = "subgraph_analysis"
    EMBEDDING_GENERATION = "embedding_generation"
    SIMILARITY_CALCULATION = "similarity_calculation"


@dataclass
class JobConfiguration:
    """Job configuration model"""

    min_density: Optional[float] = None
    prompt: Optional[str] = None
    hyperparameters: Dict[str, Any] = None
    analysis_type: str = "default"
    custom_settings: Dict[str, Any] = None

    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.custom_settings is None:
            self.custom_settings = {}

    def to_dict(self) -> Dict:
        return {
            "min_density": self.min_density,
            "prompt": self.prompt,
            "hyperparameters": self.hyperparameters,
            "analysis_type": self.analysis_type,
            "custom_settings": self.custom_settings,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "JobConfiguration":
        return cls(
            min_density=data.get("min_density"),
            prompt=data.get("prompt"),
            hyperparameters=data.get("hyperparameters", {}),
            analysis_type=data.get("analysis_type", "default"),
            custom_settings=data.get("custom_settings", {}),
        )


@dataclass
class JobResult:
    """Job result model"""

    job_id: str
    result_data: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: List[str]
    execution_time: float
    memory_usage: Optional[int] = None
    error_details: Optional[str] = None

    def __post_init__(self):
        if not self.artifacts:
            self.artifacts = []

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "result_data": self.result_data,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "error_details": self.error_details,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "JobResult":
        return cls(
            job_id=data["job_id"],
            result_data=data["result_data"],
            metrics=data["metrics"],
            artifacts=data.get("artifacts", []),
            execution_time=data["execution_time"],
            memory_usage=data.get("memory_usage"),
            error_details=data.get("error_details"),
        )


@dataclass
class Job:
    """Job model representing analysis projects"""

    job_id: str
    user_id: str
    file_id: str
    file_version_id: Optional[str] = None
    job_type: JobType = JobType.ANALYSIS
    status: JobStatus = JobStatus.PENDING
    title: Optional[str] = None
    description: Optional[str] = None
    configuration: Optional[JobConfiguration] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime = None
    progress: float = 0.0
    result: Optional[JobResult] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    tags: List[str] = None

    def __post_init__(self):
        if not self.job_id:
            self.job_id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.configuration is None:
            self.configuration = JobConfiguration()
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "file_id": self.file_id,
            "file_version_id": self.file_version_id,
            "job_type": (
                self.job_type.value
                if isinstance(self.job_type, JobType)
                else self.job_type
            ),
            "status": (
                self.status.value if isinstance(self.status, JobStatus) else self.status
            ),
            "title": self.title,
            "description": self.description,
            "configuration": (
                self.configuration.to_dict() if self.configuration else None
            ),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "updated_at": self.updated_at.isoformat(),
            "progress": self.progress,
            "result": self.result.to_dict() if self.result else None,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Job":
        return cls(
            job_id=data["job_id"],
            user_id=data["user_id"],
            file_id=data["file_id"],
            file_version_id=data.get("file_version_id"),
            job_type=JobType(data.get("job_type", "analysis")),
            status=JobStatus(data.get("status", "pending")),
            title=data.get("title"),
            description=data.get("description"),
            configuration=(
                JobConfiguration.from_dict(data["configuration"])
                if data.get("configuration")
                else JobConfiguration()
            ),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=(
                datetime.fromisoformat(data["started_at"])
                if data.get("started_at")
                else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            progress=data.get("progress", 0.0),
            result=JobResult.from_dict(data["result"]) if data.get("result") else None,
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )

    def update_status(self, status: JobStatus, message: Optional[str] = None):
        """Update job status and timestamp"""
        self.status = status
        self.updated_at = datetime.now()

        if status == JobStatus.RUNNING and not self.started_at:
            self.started_at = datetime.now()
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            self.completed_at = datetime.now()

        if message and status == JobStatus.FAILED:
            self.error_message = message

    def update_progress(self, progress: float):
        """Update job progress"""
        self.progress = max(0.0, min(100.0, progress))
        self.updated_at = datetime.now()

    def is_finished(self) -> bool:
        """Check if job is in a finished state"""
        return self.status in [
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        ]

    def is_active(self) -> bool:
        """Check if job is actively running"""
        return self.status in [JobStatus.QUEUED, JobStatus.RUNNING]

    def get_duration(self) -> Optional[float]:
        """Get job execution duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at and not self.completed_at:
            return (datetime.now() - self.started_at).total_seconds()
        return None


@dataclass
class JobStats:
    """Job statistics model"""

    user_id: str
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    running_jobs: int = 0
    avg_execution_time: Optional[float] = None
    total_execution_time: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "total_jobs": self.total_jobs,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "running_jobs": self.running_jobs,
            "avg_execution_time": self.avg_execution_time,
            "total_execution_time": self.total_execution_time,
        }
