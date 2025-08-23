from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    min_density: Optional[float] = None


class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    message: str
    timestamp: datetime


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    timestamp: datetime
