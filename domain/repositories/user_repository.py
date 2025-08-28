#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List, Optional

from domain.entities.user import User, UserFile


class UserRepository(ABC):
    """User repository interface"""

    @abstractmethod
    def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        pass

    @abstractmethod
    def save(self, user: User) -> None:
        """Save or update user"""
        pass

    @abstractmethod
    def delete(self, user_id: str) -> bool:
        """Delete user"""
        pass

    @abstractmethod
    def list_all(self) -> List[str]:
        """List all user IDs"""
        pass

    @abstractmethod
    def get_user_files(self, user_id: str) -> List[UserFile]:
        """Get all files for user"""
        pass

    @abstractmethod
    def add_user_file(self, user_id: str, user_file: UserFile) -> None:
        """Add file to user"""
        pass

    @abstractmethod
    def remove_user_file(self, user_id: str, filename: str) -> bool:
        """Remove file from user"""
        pass

    @abstractmethod
    def update_file_analysis(self, user_id: str, filename: str) -> None:
        """Update file analysis count"""
        pass
