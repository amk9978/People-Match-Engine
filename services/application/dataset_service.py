
from typing import Dict, List, Optional

import pandas as pd

from domain.entities.dataset import (
    Dataset,
    DatasetDeletionCriteria,
    DatasetModification,
)
from domain.repositories.dataset_repository import DatasetRepository
from domain.services.dataset_modification_service import DatasetModificationService


class DatasetService:
    """Application service for dataset management"""

    def __init__(self, dataset_repository: DatasetRepository):
        self.dataset_repository = dataset_repository
        self.modification_service = DatasetModificationService(dataset_repository)

    def create_dataset_from_dataframe(
        self, user_id: str, filename: str, df: pd.DataFrame
    ) -> Dict:
        """Create new dataset from uploaded DataFrame"""
        if hasattr(self.dataset_repository, "create_from_dataframe"):
            dataset = self.dataset_repository.create_from_dataframe(
                user_id, filename, df
            )
        else:
            raise NotImplementedError("Repository must implement create_from_dataframe")

        return {
            "dataset_id": f"{user_id}_{filename}",
            "version_id": dataset.current_version_id,
            "row_count": dataset.current_row_count,
            "columns": dataset.columns,
            "created_at": dataset.created_at.isoformat(),
        }

    def get_dataset_info(self, user_id: str, filename: str) -> Optional[Dict]:
        """Get dataset metadata and version information"""
        dataset = self.dataset_repository.get_by_user_and_filename(user_id, filename)
        if not dataset:
            return None

        return dataset.to_dict()

    def get_dataset_preview(
        self,
        user_id: str,
        filename: str,
        version_id: Optional[str] = None,
        limit: int = 10,
    ) -> Optional[Dict]:
        """Get preview of dataset with limited rows"""
        df = self.dataset_repository.load_dataframe(user_id, filename, version_id)
        if df is None:
            return None

        preview_df = df.head(limit)

        return {
            "total_rows": len(df),
            "preview_rows": len(preview_df),
            "columns": df.columns.tolist(),
            "data": preview_df.to_dict("records"),
        }

    def add_rows(
        self,
        user_id: str,
        filename: str,
        rows: List[Dict],
        description: Optional[str] = None,
    ) -> Dict:
        """Add new rows to dataset"""
        modification = DatasetModification(rows=rows, description=description)
        new_version = self.modification_service.add_rows(
            user_id, filename, modification
        )

        return {
            "version_id": new_version.version_id,
            "operation": new_version.operation_type,
            "row_count": new_version.row_count,
            "changes": new_version.changes,
            "description": new_version.description,
            "created_at": new_version.created_at.isoformat(),
        }

    def delete_rows(
        self,
        user_id: str,
        filename: str,
        row_indices: List[int],
        description: Optional[str] = None,
    ) -> Dict:
        """Delete rows by index"""
        new_version = self.modification_service.delete_rows_by_indices(
            user_id, filename, row_indices, description
        )

        return {
            "version_id": new_version.version_id,
            "operation": new_version.operation_type,
            "row_count": new_version.row_count,
            "changes": new_version.changes,
            "description": new_version.description,
            "created_at": new_version.created_at.isoformat(),
        }

    def delete_rows_by_criteria(
        self,
        user_id: str,
        filename: str,
        criteria: Dict,
        description: Optional[str] = None,
    ) -> Dict:
        """Delete rows matching criteria"""
        deletion_criteria = DatasetDeletionCriteria(
            criteria=criteria, description=description
        )
        new_version = self.modification_service.delete_rows_by_criteria(
            user_id, filename, deletion_criteria
        )

        return {
            "version_id": new_version.version_id,
            "operation": new_version.operation_type,
            "row_count": new_version.row_count,
            "changes": new_version.changes,
            "description": new_version.description,
            "created_at": new_version.created_at.isoformat(),
        }

    def revert_to_version(
        self,
        user_id: str,
        filename: str,
        target_version_id: str,
        description: Optional[str] = None,
    ) -> Dict:
        """Revert dataset to specific version"""
        new_version = self.modification_service.revert_to_version(
            user_id, filename, target_version_id, description
        )

        return {
            "version_id": new_version.version_id,
            "operation": new_version.operation_type,
            "row_count": new_version.row_count,
            "changes": new_version.changes,
            "description": new_version.description,
            "created_at": new_version.created_at.isoformat(),
        }

    def get_version_diff(
        self, user_id: str, filename: str, version1: str, version2: str
    ) -> Dict:
        """Get differences between two versions"""
        dataset = self.dataset_repository.get_by_user_and_filename(user_id, filename)
        if not dataset:
            raise ValueError("Dataset not found")

        return dataset.get_version_diff(version1, version2)

    def load_dataset(
        self, user_id: str, filename: str, version_id: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Load dataset as DataFrame"""
        return self.dataset_repository.load_dataframe(user_id, filename, version_id)

    def list_user_datasets(self, user_id: str) -> List[Dict]:
        """List all datasets for a user"""
        datasets = self.dataset_repository.list_by_user(user_id)

        return [
            {
                "filename": dataset.filename,
                "current_version": dataset.current_version_id,
                "total_versions": dataset.total_versions,
                "current_rows": dataset.current_row_count,
                "created_at": dataset.created_at.isoformat(),
                "last_modified": (
                    dataset.versions[-1].created_at.isoformat()
                    if dataset.versions
                    else None
                ),
            }
            for dataset in datasets
        ]
