import hashlib
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from models.file import DatasetDiff, File, FileStats, FileVersion
from services.redis.redis_cache import RedisCache

logger = logging.getLogger(__name__)


class FileService:
    """File service with versioning and Redis operations"""

    def __init__(self):
        self.cache = RedisCache()
        self.file_prefix = "file:"
        self.file_version_prefix = "file_version:"
        self.user_files_prefix = "user_files:"
        self.files_set_key = "files_set"
        self.data_directory = "/home/ryan/PycharmProjects/match_engine/data"

        os.makedirs(self.data_directory, exist_ok=True)

    def create_file(
        self,
        user_id: str,
        filename: str,
        original_filename: str,
        df: pd.DataFrame,
        description: str = "Initial upload",
    ) -> File:
        """Create a new file with initial version"""
        file_id = str(uuid.uuid4())
        now = datetime.now()
        file_obj = File(
            file_id=file_id,
            user_id=user_id,
            filename=filename,
            original_filename=original_filename,
            created_at=now,
            updated_at=now,
            current_version_id="",  # Will be set after version creation
        )

        version = self._create_file_version(file_obj.file_id, 1, df, description)

        file_obj.current_version_id = version.version_id

        self._save_file(file_obj)

        file_path = self._save_file_data_to_disk(version.version_id, df)
        version.file_path = file_path

        self.cache.sadd(self.files_set_key, file_obj.file_id)
        self.cache.sadd(f"{self.user_files_prefix}{user_id}", file_obj.file_id)

        logger.info(f"Created file {file_obj.file_id} for user {user_id}")
        return file_obj

    def get_file(self, file_id: str) -> Optional[File]:
        """Get file by ID"""
        file_key = f"{self.file_prefix}{file_id}"
        file_data = self.cache.get(file_key)

        if not file_data:
            return None

        try:
            return File.from_dict(json.loads(file_data))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing file data for {file_id}: {e}")
            return None

    def get_file_by_name(self, user_id: str, filename: str) -> Optional[File]:
        """Get file by user and filename"""
        user_files = self.get_user_files(user_id)

        for file_obj in user_files:
            if file_obj.filename == filename:
                return file_obj

        return None

    def get_user_files(self, user_id: str) -> List[File]:
        """Get all files for a user"""
        files_key = f"{self.user_files_prefix}{user_id}"
        file_ids = self.cache.smembers(files_key)

        files = []
        for file_id in file_ids:
            file_id_str = (
                file_id.decode("utf-8") if isinstance(file_id, bytes) else file_id
            )
            file_obj = self.get_file(file_id_str)
            if file_obj:
                files.append(file_obj)

        files.sort(key=lambda f: f.created_at, reverse=True)
        return files

    def load_file_data(
        self, file_id: str, version_id: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Load file data, optionally specific version"""
        file_obj = self.get_file(file_id)
        if not file_obj:
            return None

        if not version_id:
            version_id = file_obj.current_version_id

        return self._load_file_data(version_id)

    def add_file_version(
        self,
        file_id: str,
        df: pd.DataFrame,
        description: str,
        changes_summary: Dict[str, Any],
    ) -> Optional[FileVersion]:
        """Add new version to existing file"""
        file_obj = self.get_file(file_id)
        if not file_obj:
            return None

        new_version_number = file_obj.total_versions + 1
        version = self._create_file_version(
            file_id, new_version_number, df, description, changes_summary
        )

        file_obj.current_version_id = version.version_id
        file_obj.increment_version_count()

        file_path = self._save_file_data_to_disk(version.version_id, df)
        version.file_path = file_path
        self._save_file(file_obj)

        version_key = f"{self.file_version_prefix}{version.version_id}"
        self.cache.set(version_key, json.dumps(version.to_dict()))

        logger.info(f"Added version {new_version_number} to file {file_id}")
        return version

    def get_file_version(self, version_id: str) -> Optional[FileVersion]:
        """Get file version by ID"""
        version_key = f"{self.file_version_prefix}{version_id}"
        version_data = self.cache.get(version_key)

        if not version_data:
            return None

        try:
            return FileVersion.from_dict(json.loads(version_data))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing version data for {version_id}: {e}")
            return None

    def get_file_versions(self, file_id: str) -> List[FileVersion]:
        """Get all versions for a file"""

        version_pattern = f"{self.file_version_prefix}*"
        version_keys = []

        try:
            import redis

            r = redis.from_url(self.cache.redis_url)
            all_keys = r.keys(version_pattern)

            for key in all_keys:
                key_str = key.decode("utf-8")
                version_id = key_str.replace(self.file_version_prefix, "")
                version_data = self.cache.get(key_str)

                if version_data:
                    try:
                        data = json.loads(version_data)
                        if data.get("file_id") == file_id:
                            version_keys.append(key_str)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception as e:
            logger.error(f"Error getting file versions for {file_id}: {e}")
            return []

        versions = []
        for key in version_keys:
            version_data = self.cache.get(key)
            if version_data:
                try:
                    version = FileVersion.from_dict(json.loads(version_data))
                    versions.append(version)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        versions.sort(key=lambda v: v.version_number)
        return versions

    def delete_file(self, file_id: str) -> bool:
        """Delete file and all its versions"""
        try:
            file_obj = self.get_file(file_id)
            if not file_obj:
                return False

            versions = self.get_file_versions(file_id)
            for version in versions:
                version_key = f"{self.file_version_prefix}{version.version_id}"
                self.cache.delete(version_key)

                if version.file_path and os.path.exists(version.file_path):
                    try:
                        os.remove(version.file_path)
                        logger.info(f"Deleted file data: {version.file_path}")
                    except Exception as e:
                        logger.error(
                            f"Error deleting file data {version.file_path}: {e}"
                        )

            file_key = f"{self.file_prefix}{file_id}"
            self.cache.delete(file_key)

            self.cache.srem(self.files_set_key, file_id)
            self.cache.srem(f"{self.user_files_prefix}{file_obj.user_id}", file_id)

            logger.info(f"Deleted file {file_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file {file_id}: {e}")
            return False

    def get_file_stats(self, file_id: str) -> Optional[FileStats]:
        """Get file statistics"""
        file_obj = self.get_file(file_id)
        if not file_obj:
            return None

        current_version = self.get_file_version(file_obj.current_version_id)
        current_size = current_version.file_size if current_version else 0

        return FileStats(
            file_id=file_obj.file_id,
            filename=file_obj.filename,
            user_id=file_obj.user_id,
            total_versions=file_obj.total_versions,
            total_jobs=file_obj.total_jobs,
            current_size=current_size,
            created_at=file_obj.created_at,
            last_modified=file_obj.updated_at,
        )

    def compare_versions(
        self, version1_id: str, version2_id: str
    ) -> Optional[DatasetDiff]:
        """Compare two versions of a dataset"""
        try:
            df1 = self._load_file_data(version1_id)
            df2 = self._load_file_data(version2_id)

            if df1 is None or df2 is None:
                return None

            added_rows = []
            deleted_rows = []
            modified_rows = []

            cols1 = set(df1.columns)
            cols2 = set(df2.columns)
            added_columns = list(cols2 - cols1)
            deleted_columns = list(cols1 - cols2)

            row_diff = len(df2) - len(df1)
            if row_diff > 0:
                added_rows = list(range(len(df1), len(df2)))
            elif row_diff < 0:
                deleted_rows = list(range(len(df2), len(df1)))

            summary = {
                "added_rows": len(added_rows),
                "deleted_rows": len(deleted_rows),
                "modified_rows": len(modified_rows),
                "added_columns": len(added_columns),
                "deleted_columns": len(deleted_columns),
            }

            return DatasetDiff(
                version1_id=version1_id,
                version2_id=version2_id,
                added_rows=added_rows,
                deleted_rows=deleted_rows,
                modified_rows=modified_rows,
                added_columns=added_columns,
                deleted_columns=deleted_columns,
                summary=summary,
            )
        except Exception as e:
            logger.error(
                f"Error comparing versions {version1_id} and {version2_id}: {e}"
            )
            return None

    def _create_file_version(
        self,
        file_id: str,
        version_number: int,
        df: pd.DataFrame,
        description: str,
        changes_summary: Dict[str, Any] = None,
    ) -> FileVersion:
        """Create a file version"""
        if changes_summary is None:
            changes_summary = {
                "type": "initial" if version_number == 1 else "modification"
            }

        data_hash = hashlib.md5(df.to_csv(index=False).encode()).hexdigest()

        file_size = len(df.to_csv(index=False).encode())

        version = FileVersion(
            version_id="",
            file_id=file_id,
            version_number=version_number,
            created_at=datetime.now(),
            description=description,
            changes_summary=changes_summary,
            data_hash=data_hash,
            row_count=len(df),
            column_count=len(df.columns),
            file_size=file_size,
        )

        version_key = f"{self.file_version_prefix}{version.version_id}"
        self.cache.set(version_key, json.dumps(version.to_dict()))

        return version

    def _get_file_path(self, version_id: str) -> str:
        """Generate file path for a version"""
        return os.path.join(self.data_directory, f"{version_id}.csv")

    def _save_file(self, file_obj: File) -> bool:
        """Save file metadata"""
        try:
            file_key = f"{self.file_prefix}{file_obj.file_id}"
            self.cache.set(file_key, json.dumps(file_obj.to_dict()))
            return True
        except Exception as e:
            logger.error(f"Error saving file {file_obj.file_id}: {e}")
            return False

    def _save_file_data_to_disk(
        self, version_id: str, df: pd.DataFrame
    ) -> Optional[str]:
        """Save file data to disk and return file path"""
        try:
            file_path = self._get_file_path(version_id)
            df.to_csv(file_path, index=False)
            logger.info(f"Saved file data to: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving file data for version {version_id}: {e}")
            return None

    def _load_file_data(self, version_id: str) -> Optional[pd.DataFrame]:
        """Load file data from disk"""
        try:

            version = self.get_file_version(version_id)
            if version and version.file_path and os.path.exists(version.file_path):
                return pd.read_csv(version.file_path)

            file_path = self._get_file_path(version_id)
            if os.path.exists(file_path):
                return pd.read_csv(file_path)

            logger.warning(f"File data not found for version {version_id}")
            return None
        except Exception as e:
            logger.error(f"Error loading file data for version {version_id}: {e}")
            return None

    @staticmethod
    async def save_uploaded_file(file) -> str:
        """Save uploaded file to temporary location"""
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".csv", delete=False
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            return tmp_file.name

    @staticmethod
    def validate_csv_file(filename: str) -> bool:
        """Validate if file is CSV"""
        return filename.lower().endswith(".csv")
