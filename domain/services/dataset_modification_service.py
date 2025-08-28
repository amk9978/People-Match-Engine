#!/usr/bin/env python3

from typing import Dict, List

import pandas as pd

from domain.entities.dataset import (
    Dataset,
    DatasetDeletionCriteria,
    DatasetModification,
    DatasetVersion,
)
from domain.repositories.dataset_repository import DatasetRepository


class DatasetModificationService:
    """Domain service for dataset modifications"""

    def __init__(self, dataset_repository: DatasetRepository):
        self.dataset_repository = dataset_repository

    def add_rows(
        self, user_id: str, filename: str, modification: DatasetModification
    ) -> DatasetVersion:
        """Add rows to dataset"""
        dataset = self.dataset_repository.get_by_user_and_filename(user_id, filename)
        if not dataset:
            raise ValueError("Dataset not found")

        # Validate new rows
        modification.validate_columns(dataset.columns)

        # Load current data
        current_df = self.dataset_repository.load_dataframe(user_id, filename)
        if current_df is None:
            raise ValueError("Could not load current dataset")

        # Add new rows
        if modification.rows:
            new_df = pd.DataFrame(modification.rows)
            new_df = new_df[dataset.columns]  # Ensure column order
            updated_df = pd.concat([current_df, new_df], ignore_index=True)
        else:
            updated_df = current_df

        # Create new version
        new_version = DatasetVersion.create_modification(
            operation_type="add_rows",
            row_count=len(updated_df),
            changes={"added_rows": len(modification.rows), "deleted_rows": 0},
            file_path="",  # Will be set by repository
            description=modification.description
            or f"Added {len(modification.rows)} rows",
        )

        # Save dataframe and update file path
        file_path = self.dataset_repository.save_dataframe(
            user_id, filename, new_version.version_id, updated_df
        )
        new_version.file_path = file_path

        # Update dataset
        dataset.add_version(new_version)
        self.dataset_repository.save(dataset)

        return new_version

    def delete_rows_by_indices(
        self,
        user_id: str,
        filename: str,
        row_indices: List[int],
        description: str = None,
    ) -> DatasetVersion:
        """Delete rows by indices"""
        dataset = self.dataset_repository.get_by_user_and_filename(user_id, filename)
        if not dataset:
            raise ValueError("Dataset not found")

        # Load current data
        current_df = self.dataset_repository.load_dataframe(user_id, filename)
        if current_df is None:
            raise ValueError("Could not load current dataset")

        # Validate indices
        invalid_indices = [i for i in row_indices if i < 0 or i >= len(current_df)]
        if invalid_indices:
            raise ValueError(f"Invalid row indices: {invalid_indices}")

        # Delete rows
        updated_df = current_df.drop(row_indices).reset_index(drop=True)

        # Create new version
        new_version = DatasetVersion.create_modification(
            operation_type="delete_rows",
            row_count=len(updated_df),
            changes={"added_rows": 0, "deleted_rows": len(row_indices)},
            file_path="",
            description=description or f"Deleted {len(row_indices)} rows",
        )

        # Save dataframe and update file path
        file_path = self.dataset_repository.save_dataframe(
            user_id, filename, new_version.version_id, updated_df
        )
        new_version.file_path = file_path

        # Update dataset
        dataset.add_version(new_version)
        self.dataset_repository.save(dataset)

        return new_version

    def delete_rows_by_criteria(
        self, user_id: str, filename: str, criteria: DatasetDeletionCriteria
    ) -> DatasetVersion:
        """Delete rows by criteria"""
        dataset = self.dataset_repository.get_by_user_and_filename(user_id, filename)
        if not dataset:
            raise ValueError("Dataset not found")

        # Load current data
        current_df = self.dataset_repository.load_dataframe(user_id, filename)
        if current_df is None:
            raise ValueError("Could not load current dataset")

        # Apply criteria to find rows to delete
        rows_to_delete = criteria.apply_to_dataframe(current_df)
        if rows_to_delete.empty:
            raise ValueError("No rows match the deletion criteria")

        # Delete matching rows
        updated_df = current_df.drop(rows_to_delete.index).reset_index(drop=True)
        deleted_count = len(rows_to_delete)

        # Create new version
        new_version = DatasetVersion.create_modification(
            operation_type="delete_by_criteria",
            row_count=len(updated_df),
            changes={"added_rows": 0, "deleted_rows": deleted_count},
            file_path="",
            description=criteria.description
            or f"Deleted {deleted_count} rows matching criteria",
        )

        # Save dataframe and update file path
        file_path = self.dataset_repository.save_dataframe(
            user_id, filename, new_version.version_id, updated_df
        )
        new_version.file_path = file_path

        # Update dataset
        dataset.add_version(new_version)
        self.dataset_repository.save(dataset)

        return new_version

    def revert_to_version(
        self,
        user_id: str,
        filename: str,
        target_version_id: str,
        description: str = None,
    ) -> DatasetVersion:
        """Revert dataset to specific version"""
        dataset = self.dataset_repository.get_by_user_and_filename(user_id, filename)
        if not dataset:
            raise ValueError("Dataset not found")

        # Load target version
        target_df = self.dataset_repository.load_dataframe(
            user_id, filename, target_version_id
        )
        if target_df is None:
            raise ValueError("Target version not found")

        # Create new version
        new_version = DatasetVersion.create_modification(
            operation_type="revert",
            row_count=len(target_df),
            changes={"added_rows": 0, "deleted_rows": 0},
            file_path="",
            description=description or f"Reverted to version {target_version_id}",
        )

        # Save dataframe and update file path
        file_path = self.dataset_repository.save_dataframe(
            user_id, filename, new_version.version_id, target_df
        )
        new_version.file_path = file_path

        # Update dataset
        dataset.add_version(new_version)
        self.dataset_repository.save(dataset)

        return new_version
