#!/usr/bin/env python3

import json
import time
from typing import Dict, List, Optional
from datetime import datetime

from services.redis_cache import RedisEmbeddingCache


class UserService:
    """Simple user management service using Redis"""

    def __init__(self):
        self.cache = RedisEmbeddingCache()

    def _get_user_key(self, user_id: str) -> str:
        """Generate Redis key for user data"""
        return f"user:{user_id}"

    def _get_user_files_key(self, user_id: str) -> str:
        """Generate Redis key for user's file list"""
        return f"user:{user_id}:files"

    def create_user(self, user_id: str) -> Dict:
        """Create or get user record"""
        user_key = self._get_user_key(user_id)
        
        # Check if user already exists
        existing_user = self.cache.get(user_key)
        if existing_user:
            try:
                return json.loads(existing_user)
            except json.JSONDecodeError:
                pass
        
        # Create new user
        user_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_active": datetime.utcnow().isoformat(),
            "total_files": 0,
            "total_analyses": 0
        }
        
        self.cache.set(user_key, json.dumps(user_data))
        
        # Initialize empty file list
        files_key = self._get_user_files_key(user_id)
        self.cache.set(files_key, json.dumps([]))
        
        return user_data

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user record"""
        user_key = self._get_user_key(user_id)
        user_data = self.cache.get(user_key)
        
        if user_data:
            try:
                return json.loads(user_data)
            except json.JSONDecodeError:
                return None
        return None

    def update_user_activity(self, user_id: str) -> None:
        """Update user's last activity timestamp"""
        user_data = self.get_user(user_id)
        if user_data:
            user_data["last_active"] = datetime.utcnow().isoformat()
            user_key = self._get_user_key(user_id)
            self.cache.set(user_key, json.dumps(user_data))

    def add_user_file(self, user_id: str, filename: str, file_size: int = None) -> Dict:
        """Add a file to user's file list"""
        # Ensure user exists
        user_data = self.create_user(user_id)
        
        # Get current file list
        files_key = self._get_user_files_key(user_id)
        files_data = self.cache.get(files_key)
        
        try:
            files = json.loads(files_data) if files_data else []
        except json.JSONDecodeError:
            files = []
        
        # Create file record
        file_record = {
            "filename": filename,
            "uploaded_at": datetime.utcnow().isoformat(),
            "file_size": file_size,
            "analysis_count": 0,
            "last_analysis": None
        }
        
        # Check if file already exists (replace it)
        files = [f for f in files if f["filename"] != filename]
        files.append(file_record)
        
        # Sort by upload time (newest first)
        files.sort(key=lambda x: x["uploaded_at"], reverse=True)
        
        # Update file list
        self.cache.set(files_key, json.dumps(files))
        
        # Update user stats
        user_data["total_files"] = len(files)
        user_data["last_active"] = datetime.utcnow().isoformat()
        user_key = self._get_user_key(user_id)
        self.cache.set(user_key, json.dumps(user_data))
        
        return file_record

    def get_user_files(self, user_id: str) -> List[Dict]:
        """Get list of user's uploaded files"""
        files_key = self._get_user_files_key(user_id)
        files_data = self.cache.get(files_key)
        
        if files_data:
            try:
                return json.loads(files_data)
            except json.JSONDecodeError:
                return []
        return []

    def remove_user_file(self, user_id: str, filename: str) -> bool:
        """Remove a file from user's file list"""
        files_key = self._get_user_files_key(user_id)
        files_data = self.cache.get(files_key)
        
        if not files_data:
            return False
        
        try:
            files = json.loads(files_data)
        except json.JSONDecodeError:
            return False
        
        original_count = len(files)
        files = [f for f in files if f["filename"] != filename]
        
        if len(files) < original_count:
            self.cache.set(files_key, json.dumps(files))
            
            # Update user stats
            user_data = self.get_user(user_id)
            if user_data:
                user_data["total_files"] = len(files)
                user_key = self._get_user_key(user_id)
                self.cache.set(user_key, json.dumps(user_data))
            
            return True
        
        return False

    def update_file_analysis(self, user_id: str, filename: str) -> None:
        """Update analysis count and timestamp for a file"""
        files_key = self._get_user_files_key(user_id)
        files_data = self.cache.get(files_key)
        
        if not files_data:
            return
        
        try:
            files = json.loads(files_data)
        except json.JSONDecodeError:
            return
        
        # Update the specific file
        for file_record in files:
            if file_record["filename"] == filename:
                file_record["analysis_count"] = file_record.get("analysis_count", 0) + 1
                file_record["last_analysis"] = datetime.utcnow().isoformat()
                break
        
        self.cache.set(files_key, json.dumps(files))
        
        # Update user total analyses
        user_data = self.get_user(user_id)
        if user_data:
            user_data["total_analyses"] = user_data.get("total_analyses", 0) + 1
            user_key = self._get_user_key(user_id)
            self.cache.set(user_key, json.dumps(user_data))

    def get_user_stats(self, user_id: str) -> Dict:
        """Get comprehensive user statistics"""
        user_data = self.get_user(user_id)
        if not user_data:
            return {"error": "User not found"}
        
        files = self.get_user_files(user_id)
        
        # Calculate additional stats
        total_analyses = sum(f.get("analysis_count", 0) for f in files)
        recent_files = [f for f in files[:5]]  # Last 5 files
        
        return {
            "user_id": user_id,
            "created_at": user_data.get("created_at"),
            "last_active": user_data.get("last_active"),
            "total_files": len(files),
            "total_analyses": total_analyses,
            "recent_files": recent_files,
            "most_analyzed_file": max(files, key=lambda x: x.get("analysis_count", 0))["filename"] if files else None
        }

    def list_all_users(self) -> List[str]:
        """List all user IDs (for admin purposes)"""
        try:
            # Get all keys matching user pattern
            import redis
            r = redis.from_url(self.cache.redis_url)
            user_keys = r.keys("user:*")
            
            # Extract user IDs (exclude file lists)
            user_ids = []
            for key in user_keys:
                key_str = key.decode('utf-8')
                if not key_str.endswith(':files'):
                    user_id = key_str.split(':', 1)[1]
                    user_ids.append(user_id)
            
            return sorted(user_ids)
        except Exception as e:
            print(f"Error listing users: {e}")
            return []


# Global user service instance
user_service = UserService()