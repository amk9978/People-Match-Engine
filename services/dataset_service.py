
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from services.redis_cache import RedisEmbeddingCache
from services.user_service import user_service

logger = logging.getLogger(__name__)


class DatasetService:
    """Service for managing dataset modifications and versioning"""

    def __init__(self):
        self.cache = RedisEmbeddingCache()
        self.datasets_dir = "../datasets"
        os.makedirs(self.datasets_dir, exist_ok=True)

    def _get_dataset_key(self, user_id: str, filename: str) -> str:
        """Generate Redis key for dataset metadata"""
        return f"dataset:{user_id}:{filename}"

    def _get_dataset_versions_key(self, user_id: str, filename: str) -> str:
        """Generate Redis key for dataset versions list"""
        return f"dataset:{user_id}:{filename}:versions"

    def _get_dataset_path(
        self, user_id: str, filename: str, version_id: str = None
    ) -> str:
        """Generate file path for dataset storage"""
        if version_id:
            name_without_ext = os.path.splitext(filename)[0]
            return os.path.join(
                self.datasets_dir, f"{user_id}_{name_without_ext}_{version_id}.csv"
            )
        else:
            name_without_ext = os.path.splitext(filename)[0]
            return os.path.join(
                self.datasets_dir, f"{user_id}_{name_without_ext}_original.csv"
            )

    def store_original_dataset(
        self, user_id: str, filename: str, df: pd.DataFrame
    ) -> Dict:
        """Store original dataset and create version tracking"""

        # Generate version ID
        version_id = str(uuid.uuid4())[:8]

        # Store dataset file
        file_path = self._get_dataset_path(user_id, filename)
        df.to_csv(file_path, index=False)

        # Create dataset metadata
        dataset_metadata = {
            "user_id": user_id,
            "filename": filename,
            "original_version_id": version_id,
            "current_version_id": version_id,
            "created_at": datetime.utcnow().isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist(),
            "file_size": os.path.getsize(file_path),
        }

        # Store metadata in Redis
        dataset_key = self._get_dataset_key(user_id, filename)
        self.cache.set(dataset_key, json.dumps(dataset_metadata))

        # Create version record
        version_record = {
            "version_id": version_id,
            "type": "original",
            "created_at": datetime.utcnow().isoformat(),
            "row_count": len(df),
            "changes": {"added_rows": 0, "deleted_rows": 0},
            "file_path": file_path,
            "description": "Original uploaded dataset",
        }

        # Store version list
        versions_key = self._get_dataset_versions_key(user_id, filename)
        self.cache.set(versions_key, json.dumps([version_record]))

        return {
            "dataset_id": f"{user_id}_{filename}",
            "version_id": version_id,
            "row_count": len(df),
            "columns": df.columns.tolist(),
            "created_at": version_record["created_at"],
        }

    def get_dataset_info(self, user_id: str, filename: str) -> Optional[Dict]:
        """Get dataset metadata and version information"""
        dataset_key = self._get_dataset_key(user_id, filename)
        metadata = self.cache.get(dataset_key)

        if not metadata:
            return None

        try:
            dataset_info = json.loads(metadata)

            # Get version history
            versions_key = self._get_dataset_versions_key(user_id, filename)
            versions_data = self.cache.get(versions_key)
            versions = json.loads(versions_data) if versions_data else []

            dataset_info["versions"] = versions
            dataset_info["total_versions"] = len(versions)

            return dataset_info
        except json.JSONDecodeError:
            return None

    def load_dataset(
        self, user_id: str, filename: str, version_id: str = None
    ) -> Optional[pd.DataFrame]:
        """Load dataset by user, filename, and optional version"""
        dataset_info = self.get_dataset_info(user_id, filename)
        if not dataset_info:
            return None

        # Use current version if none specified
        if not version_id:
            version_id = dataset_info["current_version_id"]

        # Find the version
        versions = dataset_info.get("versions", [])
        version_record = None
        for v in versions:
            if v["version_id"] == version_id:
                version_record = v
                break

        if not version_record:
            return None

        # Load the dataset file
        try:
            return pd.read_csv(version_record["file_path"])
        except Exception as e:
            logger.info(f"Error loading dataset: {e}")
            return None

    def add_rows(
        self, user_id: str, filename: str, new_rows: List[Dict], description: str = None
    ) -> Dict:
        """Add new rows to dataset and create new version"""

        # Load current dataset
        current_df = self.load_dataset(user_id, filename)
        if current_df is None:
            raise ValueError("Dataset not found")

        # Validate new rows have same columns
        if new_rows:
            new_df = pd.DataFrame(new_rows)

            # Check columns match
            missing_cols = set(current_df.columns) - set(new_df.columns)
            if missing_cols:
                raise ValueError(f"New rows missing columns: {missing_cols}")

            extra_cols = set(new_df.columns) - set(current_df.columns)
            if extra_cols:
                raise ValueError(f"New rows have extra columns: {extra_cols}")

            # Reorder columns to match original
            new_df = new_df[current_df.columns]

            # Combine datasets
            updated_df = pd.concat([current_df, new_df], ignore_index=True)
        else:
            updated_df = current_df

        return self._create_new_version(
            user_id,
            filename,
            updated_df,
            "add_rows",
            {"added_rows": len(new_rows), "deleted_rows": 0},
            description or f"Added {len(new_rows)} rows",
        )

    def delete_rows(
        self,
        user_id: str,
        filename: str,
        row_indices: List[int],
        description: str = None,
    ) -> Dict:
        """Delete rows by index and create new version"""

        # Load current dataset
        current_df = self.load_dataset(user_id, filename)
        if current_df is None:
            raise ValueError("Dataset not found")

        # Validate row indices
        invalid_indices = [i for i in row_indices if i < 0 or i >= len(current_df)]
        if invalid_indices:
            raise ValueError(f"Invalid row indices: {invalid_indices}")

        # Delete rows
        updated_df = current_df.drop(row_indices).reset_index(drop=True)

        return self._create_new_version(
            user_id,
            filename,
            updated_df,
            "delete_rows",
            {"added_rows": 0, "deleted_rows": len(row_indices)},
            description or f"Deleted {len(row_indices)} rows",
        )

    def delete_rows_by_criteria(
        self, user_id: str, filename: str, criteria: Dict, description: str = None
    ) -> Dict:
        """Delete rows matching criteria and create new version"""

        # Load current dataset
        current_df = self.load_dataset(user_id, filename)
        if current_df is None:
            raise ValueError("Dataset not found")

        # Apply criteria to find rows to delete
        mask = pd.Series([True] * len(current_df), index=current_df.index)

        for column, value in criteria.items():
            if column not in current_df.columns:
                raise ValueError(f"Column '{column}' not found in dataset")

            if isinstance(value, dict):
                # Handle operators like {"operator": "contains", "value": "text"}
                operator = value.get("operator", "equals")
                search_value = value.get("value")

                if operator == "equals":
                    mask = mask & (current_df[column] == search_value)
                elif operator == "contains":
                    mask = mask & current_df[column].astype(str).str.contains(
                        str(search_value), na=False
                    )
                elif operator == "not_equals":
                    mask = mask & (current_df[column] != search_value)
                elif operator == "empty":
                    mask = mask & (
                        current_df[column].isna()
                        | (current_df[column].astype(str).str.strip() == "")
                    )
                elif operator == "not_empty":
                    mask = mask & (
                        current_df[column].notna()
                        & (current_df[column].astype(str).str.strip() != "")
                    )
            else:
                # Simple equality match
                mask = mask & (current_df[column] == value)

        # Find rows to delete
        rows_to_delete = current_df[mask].index.tolist()

        if not rows_to_delete:
            raise ValueError("No rows match the deletion criteria")

        # Delete matching rows
        updated_df = current_df[~mask].reset_index(drop=True)
        deleted_count = len(rows_to_delete)

        return self._create_new_version(
            user_id,
            filename,
            updated_df,
            "delete_by_criteria",
            {"added_rows": 0, "deleted_rows": deleted_count},
            description or f"Deleted {deleted_count} rows matching criteria",
        )

    def _create_new_version(
        self,
        user_id: str,
        filename: str,
        df: pd.DataFrame,
        operation_type: str,
        changes: Dict,
        description: str,
    ) -> Dict:
        """Create a new version of the dataset"""

        # Generate new version ID
        version_id = str(uuid.uuid4())[:8]

        # Save new dataset file
        file_path = self._get_dataset_path(user_id, filename, version_id)
        df.to_csv(file_path, index=False)

        # Create version record
        version_record = {
            "version_id": version_id,
            "type": operation_type,
            "created_at": datetime.utcnow().isoformat(),
            "row_count": len(df),
            "changes": changes,
            "file_path": file_path,
            "description": description,
        }

        # Update versions list
        versions_key = self._get_dataset_versions_key(user_id, filename)
        versions_data = self.cache.get(versions_key)
        versions = json.loads(versions_data) if versions_data else []
        versions.append(version_record)
        self.cache.set(versions_key, json.dumps(versions))

        # Update dataset metadata
        dataset_key = self._get_dataset_key(user_id, filename)
        metadata = json.loads(self.cache.get(dataset_key))
        metadata["current_version_id"] = version_id
        metadata["row_count"] = len(df)
        metadata["file_size"] = os.path.getsize(file_path)
        self.cache.set(dataset_key, json.dumps(metadata))

        return {
            "version_id": version_id,
            "operation": operation_type,
            "row_count": len(df),
            "changes": changes,
            "description": description,
            "created_at": version_record["created_at"],
        }

    def get_dataset_preview(
        self, user_id: str, filename: str, version_id: str = None, limit: int = 10
    ) -> Optional[Dict]:
        """Get preview of dataset with limited rows"""
        df = self.load_dataset(user_id, filename, version_id)
        if df is None:
            return None

        preview_df = df.head(limit)

        return {
            "total_rows": len(df),
            "preview_rows": len(preview_df),
            "columns": df.columns.tolist(),
            "data": preview_df.to_dict("records"),
        }

    def revert_to_version(
        self,
        user_id: str,
        filename: str,
        target_version_id: str,
        description: str = None,
    ) -> Dict:
        """Revert dataset to a specific version (creates new version)"""

        # Load the target version
        target_df = self.load_dataset(user_id, filename, target_version_id)
        if target_df is None:
            raise ValueError("Target version not found")

        return self._create_new_version(
            user_id,
            filename,
            target_df,
            "revert",
            {"added_rows": 0, "deleted_rows": 0},
            description or f"Reverted to version {target_version_id}",
        )

    def get_version_diff(
        self, user_id: str, filename: str, version1: str, version2: str
    ) -> Dict:
        """Get differences between two versions"""

        df1 = self.load_dataset(user_id, filename, version1)
        df2 = self.load_dataset(user_id, filename, version2)

        if df1 is None or df2 is None:
            raise ValueError("One or both versions not found")

        return {
            "version1": {"id": version1, "rows": len(df1)},
            "version2": {"id": version2, "rows": len(df2)},
            "row_difference": len(df2) - len(df1),
            "columns_changed": list(set(df1.columns) ^ set(df2.columns)),
            "summary": f"Version {version2} has {len(df2) - len(df1):+d} rows compared to {version1}",
        }

    def list_user_datasets(self, user_id: str) -> List[Dict]:
        """List all datasets for a user"""
        user_files = user_service.get_user_files(user_id)
        datasets = []

        for file_info in user_files:
            dataset_info = self.get_dataset_info(user_id, file_info["filename"])
            if dataset_info:
                datasets.append(
                    {
                        "filename": file_info["filename"],
                        "current_version": dataset_info["current_version_id"],
                        "total_versions": dataset_info["total_versions"],
                        "current_rows": dataset_info["row_count"],
                        "created_at": dataset_info["created_at"],
                        "last_modified": (
                            dataset_info["versions"][-1]["created_at"]
                            if dataset_info["versions"]
                            else None
                        ),
                    }
                )

        return datasets


# Global dataset service instance
dataset_service = DatasetService()
