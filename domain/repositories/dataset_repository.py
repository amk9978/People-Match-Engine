#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
from domain.entities.dataset import Dataset, DatasetVersion


class DatasetRepository(ABC):
    """Dataset repository interface"""
    
    @abstractmethod
    def get_by_user_and_filename(self, user_id: str, filename: str) -> Optional[Dataset]:
        """Get dataset by user ID and filename"""
        pass
    
    @abstractmethod
    def save(self, dataset: Dataset) -> None:
        """Save or update dataset"""
        pass
    
    @abstractmethod
    def delete(self, user_id: str, filename: str) -> bool:
        """Delete dataset"""
        pass
    
    @abstractmethod
    def list_by_user(self, user_id: str) -> List[Dataset]:
        """List all datasets for user"""
        pass
    
    @abstractmethod
    def load_dataframe(self, user_id: str, filename: str, version_id: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load dataset as DataFrame"""
        pass
    
    @abstractmethod
    def save_dataframe(self, user_id: str, filename: str, version_id: str, df: pd.DataFrame) -> str:
        """Save DataFrame and return file path"""
        pass