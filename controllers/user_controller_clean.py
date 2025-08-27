#!/usr/bin/env python3

from fastapi import HTTPException
from typing import Dict, List
from services.user_service_clean import UserService


class UserController:
    """Clean user controller with presentation layer concerns"""
    
    def __init__(self):
        self.user_service = UserService()
    
    async def get_user_stats(self, user_id: str) -> Dict:
        """Get user statistics"""
        try:
            self.user_service.update_user_activity(user_id)
            stats = self.user_service.get_user_stats(user_id)
            
            if "error" in stats:
                raise HTTPException(status_code=404, detail="User not found")
            
            return stats
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get user stats: {str(e)}")
    
    async def list_users(self) -> List[str]:
        """List all users"""
        try:
            return self.user_service.list_all_users()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list users: {str(e)}")
    
    async def get_user_files(self, user_id: str) -> List[Dict]:
        """Get user's files"""
        try:
            self.user_service.update_user_activity(user_id)
            files = self.user_service.get_user_files(user_id)
            return [f.to_dict() for f in files]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get user files: {str(e)}")
    
    async def remove_user_file(self, user_id: str, filename: str) -> Dict:
        """Remove user file"""
        try:
            success = self.user_service.remove_user_file(user_id, filename)
            if not success:
                raise HTTPException(status_code=404, detail="File not found")
            
            return {"message": f"File '{filename}' removed successfully"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to remove file: {str(e)}")