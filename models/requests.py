#!/usr/bin/env python3

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class AddRowsRequest(BaseModel):
    """Request model for adding rows"""
    rows: List[Dict] = Field(..., description="List of row data dictionaries")
    description: Optional[str] = Field(None, description="Description of the modification")


class DeleteRowsRequest(BaseModel):
    """Request model for deleting rows by index"""
    row_indices: List[int] = Field(..., description="List of row indices to delete")
    description: Optional[str] = Field(None, description="Description of the deletion")


class DeleteRowsByCriteriaRequest(BaseModel):
    """Request model for deleting rows by criteria"""
    criteria: Dict = Field(..., description="Criteria for row deletion")
    description: Optional[str] = Field(None, description="Description of the deletion")