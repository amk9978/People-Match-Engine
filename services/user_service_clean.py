#!/usr/bin/env python3

import json
from datetime import datetime
from typing import Dict, List, Optional
from models.user import User, UserFile
from services.redis_cache import RedisEmbeddingCache


class UserService:
    """Clean user service with proper separation of concerns"""
    
    def __init__(self):
        self.cache = RedisEmbeddingCache()
    
    def create_or_get_user(self, user_id: str) -> User:
        """Create user if doesn't exist, otherwise return existing"""
        user = self._get_user_from_cache(user_id)
        
        if not user:
            user = User(
                user_id=user_id,
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow(),
                total_files=0,
                total_analyses=0
            )
            self._save_user_to_cache(user)
        
        return user
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get comprehensive user statistics"""
        user = self._get_user_from_cache(user_id)
        if not user:
            return {"error": "User not found"}
        
        files = self.get_user_files(user_id)
        total_analyses = sum(f.analysis_count for f in files)
        recent_files = sorted(files, key=lambda x: x.uploaded_at, reverse=True)[:5]
        
        most_analyzed_file = None
        if files:
            most_analyzed = max(files, key=lambda x: x.analysis_count)
            if most_analyzed.analysis_count > 0:
                most_analyzed_file = most_analyzed.filename
        
        return {
            "user_id": user.user_id,
            "created_at": user.created_at.isoformat(),
            "last_active": user.last_active.isoformat(),
            "total_files": len(files),
            "total_analyses": total_analyses,
            "recent_files": [f.to_dict() for f in recent_files],
            "most_analyzed_file": most_analyzed_file
        }
    
    def update_user_activity(self, user_id: str) -> None:
        """Update user's last activity"""
        user = self._get_user_from_cache(user_id)
        if user:
            user.last_active = datetime.utcnow()
            self._save_user_to_cache(user)
    
    def add_user_file(self, user_id: str, filename: str, file_size: int) -> UserFile:
        """Add file to user's collection"""
        # Ensure user exists
        self.create_or_get_user(user_id)
        
        user_file = UserFile(
            filename=filename,
            uploaded_at=datetime.utcnow(),
            file_size=file_size
        )
        
        files = self.get_user_files(user_id)
        files = [f for f in files if f.filename != filename]  # Remove existing
        files.append(user_file)
        files.sort(key=lambda x: x.uploaded_at, reverse=True)
        
        self._save_user_files_to_cache(user_id, files)
        self._update_user_file_count(user_id, len(files))
        
        return user_file
    
    def get_user_files(self, user_id: str) -> List[UserFile]:
        """Get user's files"""
        files_key = f"user:{user_id}:files"
        files_data = self.cache.get(files_key)
        
        if not files_data:
            return []
        
        try:
            files_list = json.loads(files_data)
            return [
                UserFile(
                    filename=f["filename"],
                    uploaded_at=datetime.fromisoformat(f["uploaded_at"]),
                    file_size=f["file_size"],
                    analysis_count=f.get("analysis_count", 0),
                    last_analysis=datetime.fromisoformat(f["last_analysis"]) if f.get("last_analysis") else None
                )
                for f in files_list
            ]
        except (json.JSONDecodeError, KeyError, ValueError):
            return []
    
    def remove_user_file(self, user_id: str, filename: str) -> bool:
        """Remove file from user's collection"""
        files = self.get_user_files(user_id)
        original_count = len(files)
        
        files = [f for f in files if f.filename != filename]
        
        if len(files) < original_count:
            self._save_user_files_to_cache(user_id, files)
            self._update_user_file_count(user_id, len(files))
            return True
        
        return False
    
    def update_file_analysis(self, user_id: str, filename: str) -> None:
        """Update analysis count for a file"""
        files = self.get_user_files(user_id)
        
        for user_file in files:
            if user_file.filename == filename:
                user_file.analysis_count += 1
                user_file.last_analysis = datetime.utcnow()
                break
        
        self._save_user_files_to_cache(user_id, files)
        
        # Update user's total analyses
        user = self._get_user_from_cache(user_id)
        if user:
            user.total_analyses += 1
            self._save_user_to_cache(user)
    
    def list_all_users(self) -> List[str]:
        """List all user IDs"""
        try:
            import redis
            r = redis.from_url(self.cache.redis_url)
            user_keys = r.keys("user:*")
            
            user_ids = []
            for key in user_keys:
                key_str = key.decode('utf-8')
                if not key_str.endswith(':files'):
                    user_id = key_str.split(':', 1)[1]
                    user_ids.append(user_id)
            
            return sorted(user_ids)
        except Exception:
            return []
    
    # Private methods for cache operations
    def _get_user_from_cache(self, user_id: str) -> Optional[User]:
        """Get user from cache"""
        user_key = f"user:{user_id}"
        user_data = self.cache.get(user_key)
        
        if not user_data:
            return None
        
        try:
            data = json.loads(user_data)
            return User(
                user_id=data["user_id"],
                created_at=datetime.fromisoformat(data["created_at"]),
                last_active=datetime.fromisoformat(data["last_active"]),
                total_files=data.get("total_files", 0),
                total_analyses=data.get("total_analyses", 0)
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    
    def _save_user_to_cache(self, user: User) -> None:
        """Save user to cache"""
        user_key = f"user:{user.user_id}"
        self.cache.set(user_key, json.dumps(user.to_dict()))
    
    def _save_user_files_to_cache(self, user_id: str, files: List[UserFile]) -> None:
        """Save user files to cache"""
        files_key = f"user:{user_id}:files"
        files_data = [f.to_dict() for f in files]
        self.cache.set(files_key, json.dumps(files_data))
    
    def _update_user_file_count(self, user_id: str, count: int) -> None:
        """Update user's file count"""
        user = self._get_user_from_cache(user_id)
        if user:
            user.total_files = count
            self._save_user_to_cache(user)