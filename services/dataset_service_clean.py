#!/usr/bin/env python3

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4
import pandas as pd
from models.dataset import Dataset, DatasetVersion
from services.redis_cache import RedisEmbeddingCache


class DatasetService:
    """Clean dataset service with proper separation of concerns"""
    
    def __init__(self, datasets_dir: str = "datasets"):
        self.cache = RedisEmbeddingCache()
        self.datasets_dir = datasets_dir
        os.makedirs(self.datasets_dir, exist_ok=True)
    
    def create_dataset_from_dataframe(self, user_id: str, filename: str, df: pd.DataFrame) -> Dict:
        """Create new dataset from DataFrame"""
        version_id = str(uuid4())[:8]
        file_path = self._get_dataset_path(user_id, filename)
        df.to_csv(file_path, index=False)
        
        # Create original version
        original_version = DatasetVersion(
            version_id=version_id,
            operation_type="original",
            created_at=datetime.utcnow(),
            row_count=len(df),
            changes={"added_rows": 0, "deleted_rows": 0},
            file_path=file_path,
            description="Original uploaded dataset"
        )
        
        # Create dataset
        dataset = Dataset(
            user_id=user_id,
            filename=filename,
            original_version_id=version_id,
            current_version_id=version_id,
            created_at=datetime.utcnow(),
            column_count=len(df.columns),
            columns=df.columns.tolist(),
            versions=[original_version]
        )
        
        self._save_dataset_to_cache(dataset)
        
        return {
            "dataset_id": f"{user_id}_{filename}",
            "version_id": version_id,
            "row_count": len(df),
            "columns": df.columns.tolist(),
            "created_at": dataset.created_at.isoformat()
        }
    
    def get_dataset_info(self, user_id: str, filename: str) -> Optional[Dict]:
        """Get dataset information"""
        dataset = self._load_dataset_from_cache(user_id, filename)
        return dataset.to_dict() if dataset else None
    
    def get_dataset_preview(self, user_id: str, filename: str, 
                          version_id: Optional[str] = None, limit: int = 10) -> Optional[Dict]:
        """Get dataset preview"""
        df = self.load_dataset(user_id, filename, version_id)
        if df is None:
            return None
        
        preview_df = df.head(limit)
        return {
            "total_rows": len(df),
            "preview_rows": len(preview_df),
            "columns": df.columns.tolist(),
            "data": preview_df.to_dict("records")
        }
    
    def add_rows(self, user_id: str, filename: str, new_rows: List[Dict], 
                description: str = None) -> Dict:
        """Add rows to dataset"""
        dataset = self._load_dataset_from_cache(user_id, filename)
        if not dataset:
            raise ValueError("Dataset not found")
        
        # Load current data
        current_df = self.load_dataset(user_id, filename)
        if current_df is None:
            raise ValueError("Could not load current dataset")
        
        # Validate and add rows
        self._validate_rows_columns(new_rows, dataset.columns)
        
        if new_rows:
            new_df = pd.DataFrame(new_rows)[dataset.columns]
            updated_df = pd.concat([current_df, new_df], ignore_index=True)
        else:
            updated_df = current_df
        
        # Create new version
        new_version = self._create_new_version(
            dataset, "add_rows", len(updated_df),
            {"added_rows": len(new_rows), "deleted_rows": 0},
            description or f"Added {len(new_rows)} rows"
        )
        
        # Save and update
        self._save_dataframe(user_id, filename, new_version.version_id, updated_df)
        dataset.versions.append(new_version)
        dataset.current_version_id = new_version.version_id
        self._save_dataset_to_cache(dataset)
        
        return self._version_to_result_dict(new_version)
    
    def delete_rows(self, user_id: str, filename: str, row_indices: List[int], 
                   description: str = None) -> Dict:
        """Delete rows by indices"""
        dataset = self._load_dataset_from_cache(user_id, filename)
        if not dataset:
            raise ValueError("Dataset not found")
        
        current_df = self.load_dataset(user_id, filename)
        if current_df is None:
            raise ValueError("Could not load current dataset")
        
        # Validate indices
        invalid_indices = [i for i in row_indices if i < 0 or i >= len(current_df)]
        if invalid_indices:
            raise ValueError(f"Invalid row indices: {invalid_indices}")
        
        # Delete rows
        updated_df = current_df.drop(row_indices).reset_index(drop=True)
        
        # Create new version
        new_version = self._create_new_version(
            dataset, "delete_rows", len(updated_df),
            {"added_rows": 0, "deleted_rows": len(row_indices)},
            description or f"Deleted {len(row_indices)} rows"
        )
        
        # Save and update
        self._save_dataframe(user_id, filename, new_version.version_id, updated_df)
        dataset.versions.append(new_version)
        dataset.current_version_id = new_version.version_id
        self._save_dataset_to_cache(dataset)
        
        return self._version_to_result_dict(new_version)
    
    def delete_rows_by_criteria(self, user_id: str, filename: str, 
                               criteria: Dict, description: str = None) -> Dict:
        """Delete rows by criteria"""
        dataset = self._load_dataset_from_cache(user_id, filename)
        if not dataset:
            raise ValueError("Dataset not found")
        
        current_df = self.load_dataset(user_id, filename)
        if current_df is None:
            raise ValueError("Could not load current dataset")
        
        # Apply criteria
        mask = self._build_criteria_mask(current_df, criteria)
        rows_to_delete = current_df[mask]
        
        if rows_to_delete.empty:
            raise ValueError("No rows match the deletion criteria")
        
        # Delete matching rows
        updated_df = current_df[~mask].reset_index(drop=True)
        deleted_count = len(rows_to_delete)
        
        # Create new version
        new_version = self._create_new_version(
            dataset, "delete_by_criteria", len(updated_df),
            {"added_rows": 0, "deleted_rows": deleted_count},
            description or f"Deleted {deleted_count} rows matching criteria"
        )
        
        # Save and update
        self._save_dataframe(user_id, filename, new_version.version_id, updated_df)
        dataset.versions.append(new_version)
        dataset.current_version_id = new_version.version_id
        self._save_dataset_to_cache(dataset)
        
        return self._version_to_result_dict(new_version)
    
    def revert_to_version(self, user_id: str, filename: str, target_version_id: str, 
                         description: str = None) -> Dict:
        """Revert to specific version"""
        dataset = self._load_dataset_from_cache(user_id, filename)
        if not dataset:
            raise ValueError("Dataset not found")
        
        target_df = self.load_dataset(user_id, filename, target_version_id)
        if target_df is None:
            raise ValueError("Target version not found")
        
        # Create new version
        new_version = self._create_new_version(
            dataset, "revert", len(target_df),
            {"added_rows": 0, "deleted_rows": 0},
            description or f"Reverted to version {target_version_id}"
        )
        
        # Save and update
        self._save_dataframe(user_id, filename, new_version.version_id, target_df)
        dataset.versions.append(new_version)
        dataset.current_version_id = new_version.version_id
        self._save_dataset_to_cache(dataset)
        
        return self._version_to_result_dict(new_version)
    
    def get_version_diff(self, user_id: str, filename: str, version1: str, version2: str) -> Dict:
        """Compare two versions"""
        dataset = self._load_dataset_from_cache(user_id, filename)
        if not dataset:
            raise ValueError("Dataset not found")
        
        v1 = dataset.get_version(version1)
        v2 = dataset.get_version(version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        return {
            "version1": {"id": version1, "rows": v1.row_count},
            "version2": {"id": version2, "rows": v2.row_count},
            "row_difference": v2.row_count - v1.row_count,
            "columns_changed": [],
            "summary": f"Version {version2} has {v2.row_count - v1.row_count:+d} rows compared to {version1}"
        }
    
    def load_dataset(self, user_id: str, filename: str, version_id: str = None) -> Optional[pd.DataFrame]:
        """Load dataset as DataFrame"""
        dataset = self._load_dataset_from_cache(user_id, filename)
        if not dataset:
            return None
        
        if not version_id:
            version_id = dataset.current_version_id
        
        version = dataset.get_version(version_id)
        if not version:
            return None
        
        try:
            return pd.read_csv(version.file_path)
        except Exception:
            return None
    
    def list_user_datasets(self, user_id: str) -> List[Dict]:
        """List all datasets for user"""
        try:
            import redis
            r = redis.from_url(self.cache.redis_url)
            pattern = f"dataset:{user_id}:*"
            keys = r.keys(pattern)
            
            datasets = []
            for key in keys:
                key_str = key.decode('utf-8')
                if not key_str.endswith(':versions'):
                    parts = key_str.split(':', 2)
                    if len(parts) >= 3:
                        filename = parts[2]
                        dataset = self._load_dataset_from_cache(user_id, filename)
                        if dataset:
                            datasets.append({
                                "filename": dataset.filename,
                                "current_version": dataset.current_version_id,
                                "total_versions": dataset.total_versions,
                                "current_rows": dataset.current_row_count,
                                "created_at": dataset.created_at.isoformat(),
                                "last_modified": dataset.versions[-1].created_at.isoformat() if dataset.versions else None
                            })
            
            return datasets
        except Exception:
            return []
    
    # Private helper methods
    def _get_dataset_path(self, user_id: str, filename: str, version_id: str = None) -> str:
        """Get file path for dataset"""
        name_without_ext = os.path.splitext(filename)[0]
        if version_id:
            return os.path.join(self.datasets_dir, f"{user_id}_{name_without_ext}_{version_id}.csv")
        else:
            return os.path.join(self.datasets_dir, f"{user_id}_{name_without_ext}_original.csv")
    
    def _validate_rows_columns(self, rows: List[Dict], expected_columns: List[str]) -> None:
        """Validate row columns match expected"""
        if not rows:
            return
        
        for i, row in enumerate(rows):
            missing_cols = set(expected_columns) - set(row.keys())
            if missing_cols:
                raise ValueError(f"Row {i} missing columns: {missing_cols}")
            
            extra_cols = set(row.keys()) - set(expected_columns)
            if extra_cols:
                raise ValueError(f"Row {i} has extra columns: {extra_cols}")
    
    def _build_criteria_mask(self, df: pd.DataFrame, criteria: Dict) -> pd.Series:
        """Build boolean mask from criteria"""
        mask = pd.Series([True] * len(df), index=df.index)
        
        for column, value in criteria.items():
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in dataset")
            
            if isinstance(value, dict):
                operator = value.get("operator", "equals")
                search_value = value.get("value")
                
                if operator == "equals":
                    mask = mask & (df[column] == search_value)
                elif operator == "contains":
                    mask = mask & df[column].astype(str).str.contains(str(search_value), na=False)
                elif operator == "not_equals":
                    mask = mask & (df[column] != search_value)
                elif operator == "empty":
                    mask = mask & (df[column].isna() | (df[column].astype(str).str.strip() == ""))
                elif operator == "not_empty":
                    mask = mask & (df[column].notna() & (df[column].astype(str).str.strip() != ""))
            else:
                mask = mask & (df[column] == value)
        
        return mask
    
    def _create_new_version(self, dataset: Dataset, operation_type: str, row_count: int, 
                          changes: Dict, description: str) -> DatasetVersion:
        """Create new dataset version"""
        version_id = str(uuid4())[:8]
        file_path = self._get_dataset_path(dataset.user_id, dataset.filename, version_id)
        
        return DatasetVersion(
            version_id=version_id,
            operation_type=operation_type,
            created_at=datetime.utcnow(),
            row_count=row_count,
            changes=changes,
            file_path=file_path,
            description=description
        )
    
    def _save_dataframe(self, user_id: str, filename: str, version_id: str, df: pd.DataFrame) -> None:
        """Save DataFrame to file"""
        file_path = self._get_dataset_path(user_id, filename, version_id)
        df.to_csv(file_path, index=False)
    
    def _version_to_result_dict(self, version: DatasetVersion) -> Dict:
        """Convert version to result dictionary"""
        return {
            "version_id": version.version_id,
            "operation": version.operation_type,
            "row_count": version.row_count,
            "changes": version.changes,
            "description": version.description,
            "created_at": version.created_at.isoformat()
        }
    
    def _save_dataset_to_cache(self, dataset: Dataset) -> None:
        """Save dataset metadata to cache"""
        dataset_key = f"dataset:{dataset.user_id}:{dataset.filename}"
        versions_key = f"dataset:{dataset.user_id}:{dataset.filename}:versions"
        
        metadata = {
            "user_id": dataset.user_id,
            "filename": dataset.filename,
            "original_version_id": dataset.original_version_id,
            "current_version_id": dataset.current_version_id,
            "created_at": dataset.created_at.isoformat(),
            "column_count": dataset.column_count,
            "columns": dataset.columns,
            "row_count": dataset.current_row_count
        }
        
        self.cache.set(dataset_key, json.dumps(metadata))
        self.cache.set(versions_key, json.dumps([v.to_dict() for v in dataset.versions]))
    
    def _load_dataset_from_cache(self, user_id: str, filename: str) -> Optional[Dataset]:
        """Load dataset from cache"""
        dataset_key = f"dataset:{user_id}:{filename}"
        versions_key = f"dataset:{user_id}:{filename}:versions"
        
        metadata = self.cache.get(dataset_key)
        versions_data = self.cache.get(versions_key)
        
        if not metadata or not versions_data:
            return None
        
        try:
            meta = json.loads(metadata)
            versions_list = json.loads(versions_data)
            
            versions = [
                DatasetVersion(
                    version_id=v["version_id"],
                    operation_type=v["type"],
                    created_at=datetime.fromisoformat(v["created_at"]),
                    row_count=v["row_count"],
                    changes=v["changes"],
                    file_path=v["file_path"],
                    description=v["description"]
                )
                for v in versions_list
            ]
            
            return Dataset(
                user_id=meta["user_id"],
                filename=meta["filename"],
                original_version_id=meta["original_version_id"],
                current_version_id=meta["current_version_id"],
                created_at=datetime.fromisoformat(meta["created_at"]),
                column_count=meta["column_count"],
                columns=meta["columns"],
                versions=versions
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None