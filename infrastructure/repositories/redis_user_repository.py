#!/usr/bin/env python3

import json
from datetime import datetime
from typing import List, Optional
from domain.entities.user import User, UserFile
from domain.repositories.user_repository import UserRepository
from services.redis_cache import RedisEmbeddingCache


class RedisUserRepository(UserRepository):
    """Redis implementation of user repository"""
    
    def __init__(self):
        self.cache = RedisEmbeddingCache()
    
    def _get_user_key(self, user_id: str) -> str:
        """Generate Redis key for user data"""
        return f"user:{user_id}"
    
    def _get_user_files_key(self, user_id: str) -> str:
        """Generate Redis key for user's file list"""
        return f"user:{user_id}:files"
    
    def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        user_key = self._get_user_key(user_id)
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
    
    def save(self, user: User) -> None:
        """Save or update user"""
        user_key = self._get_user_key(user.user_id)
        user_data = {
            "user_id": user.user_id,
            "created_at": user.created_at.isoformat(),
            "last_active": user.last_active.isoformat(),
            "total_files": user.total_files,
            "total_analyses": user.total_analyses
        }
        self.cache.set(user_key, json.dumps(user_data))
    
    def delete(self, user_id: str) -> bool:
        """Delete user"""
        user_key = self._get_user_key(user_id)
        files_key = self._get_user_files_key(user_id)
        
        # Delete both user and files data
        user_deleted = self.cache.delete(user_key)
        files_deleted = self.cache.delete(files_key)
        
        return user_deleted or files_deleted
    
    def list_all(self) -> List[str]:
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
    
    def get_user_files(self, user_id: str) -> List[UserFile]:
        """Get all files for user"""
        files_key = self._get_user_files_key(user_id)
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
    
    def add_user_file(self, user_id: str, user_file: UserFile) -> None:
        """Add file to user"""
        files = self.get_user_files(user_id)
        
        # Remove existing file with same name
        files = [f for f in files if f.filename != user_file.filename]
        files.append(user_file)
        
        # Sort by upload time (newest first)
        files.sort(key=lambda x: x.uploaded_at, reverse=True)
        
        # Save updated file list
        files_key = self._get_user_files_key(user_id)
        files_data = [f.to_dict() for f in files]
        self.cache.set(files_key, json.dumps(files_data))
        
        # Update user's total files count
        user = self.get_by_id(user_id)
        if user:
            user.total_files = len(files)
            self.save(user)
    
    def remove_user_file(self, user_id: str, filename: str) -> bool:
        """Remove file from user"""
        files = self.get_user_files(user_id)
        original_count = len(files)
        
        # Remove the file
        files = [f for f in files if f.filename != filename]
        
        if len(files) < original_count:
            # Save updated file list
            files_key = self._get_user_files_key(user_id)
            files_data = [f.to_dict() for f in files]
            self.cache.set(files_key, json.dumps(files_data))
            
            # Update user's total files count
            user = self.get_by_id(user_id)
            if user:
                user.total_files = len(files)
                self.save(user)
            
            return True
        
        return False
    
    def update_file_analysis(self, user_id: str, filename: str) -> None:
        """Update file analysis count"""
        files = self.get_user_files(user_id)
        
        # Update the specific file
        for user_file in files:
            if user_file.filename == filename:
                user_file.increment_analysis_count()
                break
        
        # Save updated file list
        files_key = self._get_user_files_key(user_id)
        files_data = [f.to_dict() for f in files]
        self.cache.set(files_key, json.dumps(files_data))
        
        # Update user's total analyses count
        user = self.get_by_id(user_id)
        if user:
            user.increment_analysis_count()
            self.save(user)