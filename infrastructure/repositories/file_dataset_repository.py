#!/usr/bin/env python3

import json
import os
from datetime import datetime
from typing import List, Optional
import pandas as pd
from domain.entities.dataset import Dataset, DatasetVersion
from domain.repositories.dataset_repository import DatasetRepository
from services.redis_cache import RedisEmbeddingCache


class FileDatasetRepository(DatasetRepository):
    """File system + Redis implementation of dataset repository"""
    
    def __init__(self, datasets_dir: str = "datasets"):
        self.cache = RedisEmbeddingCache()
        self.datasets_dir = datasets_dir
        os.makedirs(self.datasets_dir, exist_ok=True)
    
    def _get_dataset_key(self, user_id: str, filename: str) -> str:
        """Generate Redis key for dataset metadata"""
        return f"dataset:{user_id}:{filename}"
    
    def _get_dataset_versions_key(self, user_id: str, filename: str) -> str:
        """Generate Redis key for dataset versions list"""
        return f"dataset:{user_id}:{filename}:versions"
    
    def _get_dataset_path(self, user_id: str, filename: str, version_id: str = None) -> str:
        """Generate file path for dataset storage"""
        name_without_ext = os.path.splitext(filename)[0]
        if version_id:
            return os.path.join(self.datasets_dir, f"{user_id}_{name_without_ext}_{version_id}.csv")
        else:
            return os.path.join(self.datasets_dir, f"{user_id}_{name_without_ext}_original.csv")
    
    def get_by_user_and_filename(self, user_id: str, filename: str) -> Optional[Dataset]:
        """Get dataset by user ID and filename"""
        dataset_key = self._get_dataset_key(user_id, filename)
        metadata = self.cache.get(dataset_key)
        
        if not metadata:
            return None
        
        try:
            data = json.loads(metadata)
            
            # Get version history
            versions_key = self._get_dataset_versions_key(user_id, filename)
            versions_data = self.cache.get(versions_key)
            versions_list = json.loads(versions_data) if versions_data else []
            
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
                user_id=data["user_id"],
                filename=data["filename"],
                original_version_id=data["original_version_id"],
                current_version_id=data["current_version_id"],
                created_at=datetime.fromisoformat(data["created_at"]),
                column_count=data["column_count"],
                columns=data["columns"],
                versions=versions
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    
    def save(self, dataset: Dataset) -> None:
        """Save or update dataset"""
        # Save metadata
        dataset_key = self._get_dataset_key(dataset.user_id, dataset.filename)
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
        
        # Save versions
        versions_key = self._get_dataset_versions_key(dataset.user_id, dataset.filename)
        versions_data = [v.to_dict() for v in dataset.versions]
        self.cache.set(versions_key, json.dumps(versions_data))
    
    def delete(self, user_id: str, filename: str) -> bool:
        """Delete dataset"""
        dataset = self.get_by_user_and_filename(user_id, filename)
        if not dataset:
            return False
        
        # Delete all version files
        for version in dataset.versions:
            try:
                if os.path.exists(version.file_path):
                    os.unlink(version.file_path)
            except Exception:
                pass
        
        # Delete metadata from Redis
        dataset_key = self._get_dataset_key(user_id, filename)
        versions_key = self._get_dataset_versions_key(user_id, filename)
        
        self.cache.delete(dataset_key)
        self.cache.delete(versions_key)
        
        return True
    
    def list_by_user(self, user_id: str) -> List[Dataset]:
        """List all datasets for user"""
        try:
            import redis
            r = redis.from_url(self.cache.redis_url)
            pattern = f"dataset:{user_id}:*"
            keys = r.keys(pattern)
            
            datasets = []
            for key in keys:
                key_str = key.decode('utf-8')
                # Skip version keys
                if not key_str.endswith(':versions'):
                    parts = key_str.split(':')
                    if len(parts) >= 3:
                        filename = ':'.join(parts[2:])  # Handle filenames with colons
                        dataset = self.get_by_user_and_filename(user_id, filename)
                        if dataset:
                            datasets.append(dataset)
            
            return datasets
        except Exception:
            return []
    
    def load_dataframe(self, user_id: str, filename: str, version_id: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load dataset as DataFrame"""
        dataset = self.get_by_user_and_filename(user_id, filename)
        if not dataset:
            return None
        
        # Use current version if none specified
        if not version_id:
            version_id = dataset.current_version_id
        
        # Find the version
        version = dataset.get_version(version_id)
        if not version:
            return None
        
        try:
            return pd.read_csv(version.file_path)
        except Exception:
            return None
    
    def save_dataframe(self, user_id: str, filename: str, version_id: str, df: pd.DataFrame) -> str:
        """Save DataFrame and return file path"""
        file_path = self._get_dataset_path(user_id, filename, version_id)
        df.to_csv(file_path, index=False)
        return file_path
    
    def create_from_dataframe(self, user_id: str, filename: str, df: pd.DataFrame) -> Dataset:
        """Create new dataset from DataFrame"""
        # Create original version
        original_version = DatasetVersion.create_original(
            row_count=len(df),
            file_path=self._get_dataset_path(user_id, filename)
        )
        
        # Save DataFrame
        df.to_csv(original_version.file_path, index=False)
        
        # Create dataset
        dataset = Dataset(
            user_id=user_id,
            filename=filename,
            original_version_id=original_version.version_id,
            current_version_id=original_version.version_id,
            created_at=datetime.utcnow(),
            column_count=len(df.columns),
            columns=df.columns.tolist(),
            versions=[original_version]
        )
        
        # Save to repository
        self.save(dataset)
        
        return dataset