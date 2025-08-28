#!/usr/bin/env python3

from io import StringIO
from typing import Dict, List, Optional

import pandas as pd
from fastapi import HTTPException, UploadFile

from models.requests import (
    AddRowsRequest,
    DeleteRowsByCriteriaRequest,
    DeleteRowsRequest,
)
from services.dataset_service_clean import DatasetService
from services.user_service_clean import UserService


class DatasetController:
    """Clean dataset controller with presentation layer concerns"""

    def __init__(self):
        self.dataset_service = DatasetService()
        self.user_service = UserService()

    async def upload_dataset(self, user_id: str, file: UploadFile) -> Dict:
        """Upload and create new dataset"""
        try:
            if not file.filename.endswith(".csv"):
                raise HTTPException(
                    status_code=400, detail="Only CSV files are supported"
                )

            contents = await file.read()
            df = pd.read_csv(StringIO(contents.decode("utf-8")))

            if df.empty:
                raise HTTPException(status_code=400, detail="Dataset cannot be empty")

            # Create dataset
            result = self.dataset_service.create_dataset_from_dataframe(
                user_id, file.filename, df
            )

            # Update user file tracking
            self.user_service.add_user_file(user_id, file.filename, len(contents))

            return result

        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail="Invalid CSV file")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to upload dataset: {str(e)}"
            )

    async def get_dataset_info(self, user_id: str, filename: str) -> Dict:
        """Get dataset information"""
        try:
            self.user_service.update_user_activity(user_id)
            info = self.dataset_service.get_dataset_info(user_id, filename)

            if not info:
                raise HTTPException(status_code=404, detail="Dataset not found")

            return info
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to get dataset info: {str(e)}"
            )

    async def get_dataset_preview(
        self,
        user_id: str,
        filename: str,
        version_id: Optional[str] = None,
        limit: int = 10,
    ) -> Dict:
        """Get dataset preview"""
        try:
            self.user_service.update_user_activity(user_id)
            preview = self.dataset_service.get_dataset_preview(
                user_id, filename, version_id, limit
            )

            if not preview:
                raise HTTPException(status_code=404, detail="Dataset not found")

            return preview
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to get dataset preview: {str(e)}"
            )

    async def add_rows(
        self, user_id: str, filename: str, request: AddRowsRequest
    ) -> Dict:
        """Add rows to dataset"""
        try:
            self.user_service.update_user_activity(user_id)
            result = self.dataset_service.add_rows(
                user_id, filename, request.rows, request.description
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to add rows: {str(e)}")

    async def delete_rows(
        self, user_id: str, filename: str, request: DeleteRowsRequest
    ) -> Dict:
        """Delete rows by indices"""
        try:
            self.user_service.update_user_activity(user_id)
            result = self.dataset_service.delete_rows(
                user_id, filename, request.row_indices, request.description
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to delete rows: {str(e)}"
            )

    async def delete_rows_by_criteria(
        self, user_id: str, filename: str, request: DeleteRowsByCriteriaRequest
    ) -> Dict:
        """Delete rows by criteria"""
        try:
            self.user_service.update_user_activity(user_id)
            result = self.dataset_service.delete_rows_by_criteria(
                user_id, filename, request.criteria, request.description
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to delete rows by criteria: {str(e)}"
            )

    async def revert_dataset(
        self,
        user_id: str,
        filename: str,
        version_id: str,
        description: Optional[str] = None,
    ) -> Dict:
        """Revert dataset to specific version"""
        try:
            self.user_service.update_user_activity(user_id)
            result = self.dataset_service.revert_to_version(
                user_id, filename, version_id, description
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to revert dataset: {str(e)}"
            )

    async def compare_versions(
        self, user_id: str, filename: str, version1: str, version2: str
    ) -> Dict:
        """Compare two dataset versions"""
        try:
            self.user_service.update_user_activity(user_id)
            comparison = self.dataset_service.get_version_diff(
                user_id, filename, version1, version2
            )
            return comparison
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to compare versions: {str(e)}"
            )

    async def list_user_datasets(self, user_id: str) -> List[Dict]:
        """List all datasets for user"""
        try:
            self.user_service.update_user_activity(user_id)
            return self.dataset_service.list_user_datasets(user_id)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to list datasets: {str(e)}"
            )
