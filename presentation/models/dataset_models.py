#!/usr/bin/env python3

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class AddRowsRequest(BaseModel):
    """Request model for adding rows to dataset"""
    rows: List[Dict] = Field(..., description="List of row data dictionaries")
    description: Optional[str] = Field(None, description="Description of the modification")
    
    class Config:
        schema_extra = {
            "example": {
                "rows": [
                    {
                        "Person Name": "John Doe",
                        "Person Title": "Engineer",
                        "Person Company": "TechCorp",
                        "Professional Identity - Role Specification": "Software Development",
                        "Professional Identity - Experience Level": "Senior",
                        "Company Identity - Industry Classification": "Technology",
                        "Company Market - Market Traction": "Growth",
                        "Company Offering - Value Proposition": "SaaS Solutions",
                        "All Persona Titles": "Developer;Engineer"
                    }
                ],
                "description": "Added new team member"
            }
        }


class DeleteRowsRequest(BaseModel):
    """Request model for deleting rows by index"""
    row_indices: List[int] = Field(..., description="List of row indices to delete")
    description: Optional[str] = Field(None, description="Description of the deletion")
    
    class Config:
        schema_extra = {
            "example": {
                "row_indices": [0, 2, 5],
                "description": "Removed outdated entries"
            }
        }


class DeleteRowsByCriteriaRequest(BaseModel):
    """Request model for deleting rows by criteria"""
    criteria: Dict = Field(..., description="Criteria for row deletion")
    description: Optional[str] = Field(None, description="Description of the deletion")
    
    class Config:
        schema_extra = {
            "example": {
                "criteria": {
                    "Professional Identity - Experience Level": "Junior"
                },
                "description": "Removed junior level positions"
            }
        }


class DatasetInfoResponse(BaseModel):
    """Response model for dataset information"""
    user_id: str
    filename: str
    original_version_id: str
    current_version_id: str
    created_at: str
    row_count: int
    column_count: int
    columns: List[str]
    total_versions: int
    versions: List[Dict]


class DatasetPreviewResponse(BaseModel):
    """Response model for dataset preview"""
    total_rows: int
    preview_rows: int
    columns: List[str]
    data: List[Dict]


class DatasetModificationResponse(BaseModel):
    """Response model for dataset modifications"""
    message: str
    result: Dict = Field(..., description="Modification result details")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Rows added successfully",
                "result": {
                    "version_id": "abc123",
                    "operation": "add_rows",
                    "row_count": 25,
                    "changes": {"added_rows": 2, "deleted_rows": 0},
                    "description": "Added new team member",
                    "created_at": "2023-01-01T10:00:00"
                }
            }
        }


class VersionDiffResponse(BaseModel):
    """Response model for version comparison"""
    version1: Dict
    version2: Dict
    row_difference: int
    columns_changed: List[str]
    summary: str


class DatasetListResponse(BaseModel):
    """Response model for user dataset list"""
    user_id: str
    datasets: List[Dict]
    total_datasets: int