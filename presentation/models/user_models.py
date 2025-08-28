#!/usr/bin/env python3

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class UserStatsResponse(BaseModel):
    """Response model for user statistics"""

    user_id: str
    created_at: str
    last_active: str
    total_files: int
    total_analyses: int
    recent_files: List[Dict]
    most_analyzed_file: Optional[str]


class UserFilesResponse(BaseModel):
    """Response model for user files list"""

    user_id: str
    files: List[Dict]
    total_files: int


class UserListResponse(BaseModel):
    """Response model for admin user list"""

    total_users: int
    users: List[str]


class FileDeleteResponse(BaseModel):
    """Response model for file deletion"""

    message: str = Field(..., description="Deletion result message")

    class Config:
        schema_extra = {"example": {"message": "File 'data.csv' deleted successfully"}}
