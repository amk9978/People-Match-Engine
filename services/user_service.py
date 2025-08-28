import json
from datetime import datetime
from typing import List, Optional

from models.user import User, UserStats
from services.redis.redis_cache import RedisCache


class UserService:
    """Clean user service with proper separation of concerns"""

    def __init__(self):
        self.cache = RedisCache()
        self.user_stats_prefix = "user_stats:"
        self.user_files_prefix = "user_files:"
        self.user_jobs_prefix = "user_jobs:"
        self.users_set_key = "users"

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user, create if doesn't exist"""
        if self.cache.sismember(self.users_set_key, user_id):
            return User(id=user_id)
        return self.create_or_get_user(user_id)

    def create_or_get_user(self, user_id: str) -> User:
        """Create user if doesn't exist, otherwise return existing"""
        self.cache.sadd(self.users_set_key, user_id)
        return User(id=user_id)

    def create_or_get_user_stats(self, user_id: str) -> UserStats:
        """Create user stats if doesn't exist, otherwise return existing"""
        stats = self.get_user_stats(user_id)

        if not stats:
            stats = UserStats(user_id=user_id)
            self._update_user_stats(stats)
            # Add to users set
            self.cache.sadd(self.users_set_key, user_id)

        return stats

    def get_user_stats(self, user_id: str) -> Optional[UserStats]:
        """Get user statistics"""
        stats_key = f"{self.user_stats_prefix}{user_id}"
        stats_data = self.cache.get(stats_key)

        if not stats_data:
            # Create default stats if none exist
            stats = UserStats(user_id=user_id)
            self._update_user_stats(stats)
            return stats

        try:
            data = json.loads(stats_data)
            return UserStats(
                user_id=data["user_id"],
                total_files=data.get("total_files", 0),
                total_jobs=data.get("total_jobs", 0),
                total_analyses=data.get("total_analyses", 0),
                storage_used=data.get("storage_used", 0),
                last_activity=(
                    datetime.fromisoformat(data["last_activity"])
                    if data.get("last_activity")
                    else None
                ),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return UserStats(user_id=user_id)

    def update_user_activity(self, user_id: str) -> bool:
        """Update user's last activity timestamp"""
        self.create_or_get_user(user_id)
        stats = self.get_user_stats(user_id)
        stats.update_activity()
        return self._update_user_stats(stats)

    def add_user_file(self, user_id: str, file_id: str) -> bool:
        """Add file to user's file list"""
        files_key = f"{self.user_files_prefix}{user_id}"
        try:
            self.cache.sadd(files_key, file_id)
            self.increment_user_files(user_id)
            return True
        except Exception:
            return False

    def get_user_files(self, user_id: str) -> List[str]:
        """Get user's file list"""
        files_key = f"{self.user_files_prefix}{user_id}"
        file_ids = self.cache.smembers(files_key)
        return [
            fid.decode("utf-8") if isinstance(fid, bytes) else fid for fid in file_ids
        ]

    def remove_user_file(self, user_id: str, file_id: str) -> bool:
        """Remove file from user's file list"""
        files_key = f"{self.user_files_prefix}{user_id}"
        try:
            self.cache.srem(files_key, file_id)
            # Decrement file count
            stats = self.get_user_stats(user_id)
            if stats and stats.total_files > 0:
                stats.total_files -= 1
                stats.last_activity = datetime.now()
                self._update_user_stats(stats)
            return True
        except Exception:
            return False

    def update_file_analysis(self, user_id: str) -> bool:
        """Update analysis count for a file"""
        return self.increment_user_analyses(user_id)

    def list_users(self, limit: Optional[int] = None) -> List[User]:
        """List all users"""
        user_ids = self.cache.smembers(self.users_set_key)
        if not user_ids:
            return []

        # Limit results if specified
        if limit:
            user_ids = list(user_ids)[:limit]

        users = []
        for user_id in user_ids:
            user = self.get_user(
                user_id.decode("utf-8") if isinstance(user_id, bytes) else user_id
            )
            if user:
                users.append(user)

        # Sort by creation date
        users.sort(key=lambda u: u.created_at, reverse=True)
        return users

    def add_user_job(self, user_id: str, job_id: str) -> bool:
        """Add job to user's job list"""
        jobs_key = f"{self.user_jobs_prefix}{user_id}"
        try:
            self.cache.sadd(jobs_key, job_id)
            self.increment_user_jobs(user_id)
            return True
        except Exception:
            return False

    def get_user_jobs(self, user_id: str) -> List[str]:
        """Get user's job list"""
        jobs_key = f"{self.user_jobs_prefix}{user_id}"
        job_ids = self.cache.smembers(jobs_key)
        return [
            jid.decode("utf-8") if isinstance(jid, bytes) else jid for jid in job_ids
        ]

    def increment_user_files(self, user_id: str) -> bool:
        """Increment user files count"""
        stats = self.get_user_stats(user_id)
        if stats:
            stats.total_files += 1
            stats.last_activity = datetime.now()
            return self._update_user_stats(stats)
        return False

    def increment_user_jobs(self, user_id: str) -> bool:
        """Increment user jobs count"""
        stats = self.get_user_stats(user_id)
        if stats:
            stats.total_jobs += 1
            stats.last_activity = datetime.now()
            return self._update_user_stats(stats)
        return False

    def increment_user_analyses(self, user_id: str) -> bool:
        """Increment user analyses count"""
        stats = self.get_user_stats(user_id)
        if stats:
            stats.total_analyses += 1
            stats.last_activity = datetime.now()
            return self._update_user_stats(stats)
        return False

    # Private methods for cache operations
    def _update_user_stats(self, stats: UserStats) -> bool:
        """Update user statistics"""
        stats_key = f"{self.user_stats_prefix}{stats.user_id}"
        try:
            self.cache.set(stats_key, json.dumps(stats.to_dict()))
            return True
        except Exception:
            return False
