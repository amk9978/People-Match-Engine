#!/usr/bin/env python3

from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query

from presentation.models.dataset_models import (
    AddRowsRequest,
    DatasetInfoResponse,
    DatasetListResponse,
    DatasetModificationResponse,
    DatasetPreviewResponse,
    DeleteRowsByCriteriaRequest,
    DeleteRowsRequest,
    VersionDiffResponse,
)
from services.application.dataset_service import DatasetService
from services.application.user_service import UserService


class DatasetController:
    """Dataset management controller"""

    def __init__(self, dataset_service: DatasetService, user_service: UserService):
        self.dataset_service = dataset_service
        self.user_service = user_service
        self.router = APIRouter(prefix="/datasets", tags=["datasets"])
        self._register_routes()

    def _register_routes(self):
        """Register all dataset routes"""

        @self.router.get("", response_model=DatasetListResponse)
        async def list_user_datasets(user_id: str = Header(..., alias="X-User-ID")):
            """List all datasets for the current user"""
            if not user_id:
                raise HTTPException(status_code=400, detail="X-User-ID header required")

            datasets = self.dataset_service.list_user_datasets(user_id)
            return DatasetListResponse(
                user_id=user_id, datasets=datasets, total_datasets=len(datasets)
            )

        @self.router.get("/{filename}", response_model=DatasetInfoResponse)
        async def get_dataset_info(
            filename: str, user_id: str = Header(..., alias="X-User-ID")
        ):
            """Get detailed information about a specific dataset"""
            if not user_id:
                raise HTTPException(status_code=400, detail="X-User-ID header required")

            dataset_info = self.dataset_service.get_dataset_info(user_id, filename)
            if not dataset_info:
                raise HTTPException(status_code=404, detail="Dataset not found")

            return DatasetInfoResponse(**dataset_info)

        @self.router.get("/{filename}/preview", response_model=DatasetPreviewResponse)
        async def get_dataset_preview(
            filename: str,
            user_id: str = Header(..., alias="X-User-ID"),
            version_id: Optional[str] = Query(
                None, description="Version ID to preview (default: current)"
            ),
            limit: int = Query(10, description="Number of rows to preview"),
        ):
            """Get preview of dataset with sample rows"""
            if not user_id:
                raise HTTPException(status_code=400, detail="X-User-ID header required")

            preview = self.dataset_service.get_dataset_preview(
                user_id, filename, version_id, limit
            )
            if not preview:
                raise HTTPException(
                    status_code=404, detail="Dataset or version not found"
                )

            return DatasetPreviewResponse(**preview)

        @self.router.post(
            "/{filename}/add-rows", response_model=DatasetModificationResponse
        )
        async def add_rows_to_dataset(
            filename: str,
            request: AddRowsRequest,
            user_id: str = Header(..., alias="X-User-ID"),
        ):
            """Add new rows to an existing dataset"""
            if not user_id:
                raise HTTPException(status_code=400, detail="X-User-ID header required")

            try:
                result = self.dataset_service.add_rows(
                    user_id, filename, request.rows, request.description
                )

                # Update user activity
                self.user_service.update_user_activity(user_id)

                return DatasetModificationResponse(
                    message="Rows added successfully", result=result
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to add rows: {str(e)}"
                )

        @self.router.post(
            "/{filename}/delete-rows", response_model=DatasetModificationResponse
        )
        async def delete_rows_from_dataset(
            filename: str,
            request: DeleteRowsRequest,
            user_id: str = Header(..., alias="X-User-ID"),
        ):
            """Delete specific rows by index from dataset"""
            if not user_id:
                raise HTTPException(status_code=400, detail="X-User-ID header required")

            try:
                result = self.dataset_service.delete_rows(
                    user_id, filename, request.row_indices, request.description
                )

                # Update user activity
                self.user_service.update_user_activity(user_id)

                return DatasetModificationResponse(
                    message=f"Deleted {len(request.row_indices)} rows successfully",
                    result=result,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to delete rows: {str(e)}"
                )

        @self.router.post(
            "/{filename}/delete-rows-by-criteria",
            response_model=DatasetModificationResponse,
        )
        async def delete_rows_by_criteria(
            filename: str,
            request: DeleteRowsByCriteriaRequest,
            user_id: str = Header(..., alias="X-User-ID"),
        ):
            """Delete rows matching specific criteria"""
            if not user_id:
                raise HTTPException(status_code=400, detail="X-User-ID header required")

            try:
                result = self.dataset_service.delete_rows_by_criteria(
                    user_id, filename, request.criteria, request.description
                )

                # Update user activity
                self.user_service.update_user_activity(user_id)

                return DatasetModificationResponse(
                    message=f"Deleted {result['changes']['deleted_rows']} rows matching criteria",
                    result=result,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to delete rows: {str(e)}"
                )

        @self.router.post(
            "/{filename}/revert/{version_id}",
            response_model=DatasetModificationResponse,
        )
        async def revert_dataset_to_version(
            filename: str,
            version_id: str,
            user_id: str = Header(..., alias="X-User-ID"),
            description: Optional[str] = Query(
                None, description="Description of the revert operation"
            ),
        ):
            """Revert dataset to a specific version"""
            if not user_id:
                raise HTTPException(status_code=400, detail="X-User-ID header required")

            try:
                result = self.dataset_service.revert_to_version(
                    user_id, filename, version_id, description
                )

                # Update user activity
                self.user_service.update_user_activity(user_id)

                return DatasetModificationResponse(
                    message=f"Successfully reverted to version {version_id}",
                    result=result,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to revert dataset: {str(e)}"
                )

        @self.router.get("/{filename}/diff", response_model=VersionDiffResponse)
        async def get_version_diff(
            filename: str,
            version1: str = Query(..., description="First version ID"),
            version2: str = Query(..., description="Second version ID"),
            user_id: str = Header(..., alias="X-User-ID"),
        ):
            """Get differences between two versions of a dataset"""
            if not user_id:
                raise HTTPException(status_code=400, detail="X-User-ID header required")

            try:
                diff = self.dataset_service.get_version_diff(
                    user_id, filename, version1, version2
                )
                return VersionDiffResponse(**diff)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to get version diff: {str(e)}"
                )
