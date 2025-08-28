#!/usr/bin/env python3

from fastapi import APIRouter, Header, HTTPException

from presentation.models.user_models import (
    FileDeleteResponse,
    UserFilesResponse,
    UserListResponse,
    UserStatsResponse,
)
from services.application.user_service import UserService


class UserController:
    """User management controller"""

    def __init__(self, user_service: UserService):
        self.user_service = user_service
        self.router = APIRouter(prefix="/users", tags=["users"])
        self._register_routes()

    def _register_routes(self):
        """Register all user routes"""

        @self.router.get("/me", response_model=UserStatsResponse)
        async def get_current_user(user_id: str = Header(..., alias="X-User-ID")):
            """Get current user profile and stats"""
            if not user_id:
                raise HTTPException(status_code=400, detail="X-User-ID header required")

            # Ensure user exists
            self.user_service.create_or_get_user(user_id)
            stats = self.user_service.get_user_stats(user_id)

            if "error" in stats:
                raise HTTPException(status_code=404, detail=stats["error"])

            return UserStatsResponse(**stats)

        @self.router.get("/me/files", response_model=UserFilesResponse)
        async def get_user_files(user_id: str = Header(..., alias="X-User-ID")):
            """Get user's uploaded files"""
            if not user_id:
                raise HTTPException(status_code=400, detail="X-User-ID header required")

            # Ensure user exists and update activity
            self.user_service.create_or_get_user(user_id)
            self.user_service.update_user_activity(user_id)

            files = self.user_service.get_user_files(user_id)
            files_dict = [file.to_dict() for file in files]

            return UserFilesResponse(
                user_id=user_id, files=files_dict, total_files=len(files_dict)
            )

        @self.router.delete("/me/files/{filename}", response_model=FileDeleteResponse)
        async def delete_user_file(
            filename: str, user_id: str = Header(..., alias="X-User-ID")
        ):
            """Delete a user's uploaded file"""
            if not user_id:
                raise HTTPException(status_code=400, detail="X-User-ID header required")

            success = self.user_service.remove_user_file(user_id, filename)
            if not success:
                raise HTTPException(status_code=404, detail="File not found")

            return FileDeleteResponse(message=f"File '{filename}' deleted successfully")


class AdminController:
    """Admin controller for user management"""

    def __init__(self, user_service: UserService):
        self.user_service = user_service
        self.router = APIRouter(prefix="/admin", tags=["admin"])
        self._register_routes()

    def _register_routes(self):
        """Register admin routes"""

        @self.router.get("/users", response_model=UserListResponse)
        async def list_all_users():
            """Admin endpoint to list all users"""
            users = self.user_service.list_all_users()
            return UserListResponse(total_users=len(users), users=users)
