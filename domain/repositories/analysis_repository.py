#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List, Optional
from domain.entities.analysis import AnalysisJob, AnalysisResult


class AnalysisRepository(ABC):
    """Analysis repository interface"""
    
    @abstractmethod
    def save_job(self, job: AnalysisJob) -> None:
        """Save or update analysis job"""
        pass
    
    @abstractmethod
    def get_job(self, job_id: str) -> Optional[AnalysisJob]:
        """Get analysis job by ID"""
        pass
    
    @abstractmethod
    def list_jobs_by_user(self, user_id: str, limit: int = 100) -> List[AnalysisJob]:
        """List analysis jobs for user"""
        pass
    
    @abstractmethod
    def delete_job(self, job_id: str) -> bool:
        """Delete analysis job"""
        pass
    
    @abstractmethod
    def save_result(self, result: AnalysisResult) -> None:
        """Save analysis result"""
        pass
    
    @abstractmethod
    def get_result(self, job_id: str) -> Optional[AnalysisResult]:
        """Get analysis result by job ID"""
        pass
    
    @abstractmethod
    def list_results_by_user(self, user_id: str, limit: int = 100) -> List[AnalysisResult]:
        """List analysis results for user"""
        pass