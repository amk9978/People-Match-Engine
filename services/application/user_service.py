#!/usr/bin/env python3

from datetime import datetime
from typing import Dict, List, Optional
from domain.entities.user import User, UserFile
from domain.repositories.user_repository import UserRepository


class UserService:
    """Application service for user management"""
    
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository
    
    def create_or_get_user(self, user_id: str) -> User:
        """Create user if doesn't exist, otherwise return existing"""
        user = self.user_repository.get_by_id(user_id)
        
        if not user:
            user = User(
                user_id=user_id,
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow(),
                total_files=0,
                total_analyses=0
            )
            self.user_repository.save(user)
        
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.user_repository.get_by_id(user_id)
    
    def update_user_activity(self, user_id: str) -> None:
        """Update user's last activity timestamp"""
        user = self.user_repository.get_by_id(user_id)
        if user:
            user.update_activity()
            self.user_repository.save(user)
    
    def add_user_file(self, user_id: str, filename: str, file_size: int) -> UserFile:
        """Add file to user's file list"""
        # Ensure user exists
        self.create_or_get_user(user_id)
        
        # Create file record
        user_file = UserFile(
            filename=filename,
            uploaded_at=datetime.utcnow(),
            file_size=file_size
        )
        
        # Add to repository
        self.user_repository.add_user_file(user_id, user_file)
        
        return user_file
    
    def get_user_files(self, user_id: str) -> List[UserFile]:
        """Get user's uploaded files"""
        return self.user_repository.get_user_files(user_id)
    
    def remove_user_file(self, user_id: str, filename: str) -> bool:
        """Remove file from user's file list"""
        return self.user_repository.remove_user_file(user_id, filename)
    
    def update_file_analysis(self, user_id: str, filename: str) -> None:
        """Update analysis count for a file"""
        self.user_repository.update_file_analysis(user_id, filename)
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get comprehensive user statistics"""
        user = self.user_repository.get_by_id(user_id)
        if not user:
            return {"error": "User not found"}
        
        files = self.get_user_files(user_id)
        
        # Calculate additional stats
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
    
    def list_all_users(self) -> List[str]:
        """List all user IDs"""
        return self.user_repository.list_all()