#!/usr/bin/env python3

import unittest
import pandas as pd
import tempfile
import shutil
from datetime import datetime
from infrastructure.repositories.redis_user_repository import RedisUserRepository
from infrastructure.repositories.file_dataset_repository import FileDatasetRepository
from services.application.user_service import UserService
from services.application.dataset_service import DatasetService


class TestCleanArchitecture(unittest.TestCase):
    """Integration tests for the clean architecture implementation"""
    
    def setUp(self):
        # Setup temporary directory for file repository
        self.temp_dir = tempfile.mkdtemp()
        
        # Setup repositories
        self.user_repository = RedisUserRepository()
        self.dataset_repository = FileDatasetRepository(self.temp_dir)
        
        # Setup application services
        self.user_service = UserService(self.user_repository)
        self.dataset_service = DatasetService(self.dataset_repository)
        
        # Test data
        self.user_id = "test_architecture_user"
        self.filename = "test_data.csv"
        self.test_df = pd.DataFrame({
            "Person Name": ["Alice", "Bob", "Charlie"],
            "Person Title": ["Engineer", "Manager", "Designer"],
            "Company": ["TechCorp", "BusinessCorp", "DesignCorp"]
        })
    
    def tearDown(self):
        # Cleanup temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Cleanup Redis (optional - depends on your test strategy)
        try:
            self.user_repository.delete(self.user_id)
            self.dataset_repository.delete(self.user_id, self.filename)
        except:
            pass
    
    def test_full_user_workflow(self):
        """Test complete user management workflow"""
        # 1. Create user
        user = self.user_service.create_or_get_user(self.user_id)
        self.assertEqual(user.user_id, self.user_id)
        self.assertEqual(user.total_files, 0)
        
        # 2. Add file
        user_file = self.user_service.add_user_file(self.user_id, self.filename, 1024)
        self.assertEqual(user_file.filename, self.filename)
        
        # 3. Get user files
        files = self.user_service.get_user_files(self.user_id)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].filename, self.filename)
        
        # 4. Update activity
        original_time = user.last_active
        self.user_service.update_user_activity(self.user_id)
        
        updated_user = self.user_service.get_user(self.user_id)
        self.assertGreater(updated_user.last_active, original_time)
        
        # 5. Get user stats
        stats = self.user_service.get_user_stats(self.user_id)
        self.assertEqual(stats["total_files"], 1)
        self.assertEqual(len(stats["recent_files"]), 1)
    
    def test_full_dataset_workflow(self):
        """Test complete dataset management workflow"""
        # 1. Create dataset from DataFrame
        dataset_info = self.dataset_service.create_dataset_from_dataframe(
            self.user_id, self.filename, self.test_df
        )
        self.assertEqual(dataset_info["row_count"], 3)
        self.assertEqual(len(dataset_info["columns"]), 3)
        
        # 2. Get dataset info
        info = self.dataset_service.get_dataset_info(self.user_id, self.filename)
        self.assertIsNotNone(info)
        self.assertEqual(info["row_count"], 3)
        self.assertEqual(info["total_versions"], 1)
        
        # 3. Preview dataset
        preview = self.dataset_service.get_dataset_preview(self.user_id, self.filename, limit=2)
        self.assertIsNotNone(preview)
        self.assertEqual(preview["total_rows"], 3)
        self.assertEqual(preview["preview_rows"], 2)
        
        # 4. Add rows
        new_rows = [{
            "Person Name": "David",
            "Person Title": "Analyst", 
            "Company": "DataCorp"
        }]
        add_result = self.dataset_service.add_rows(self.user_id, self.filename, new_rows, "Added David")
        self.assertEqual(add_result["row_count"], 4)
        self.assertEqual(add_result["changes"]["added_rows"], 1)
        
        # 5. Delete rows by criteria
        delete_result = self.dataset_service.delete_rows_by_criteria(
            self.user_id, self.filename, {"Person Title": "Manager"}, "Removed managers"
        )
        self.assertEqual(delete_result["row_count"], 3)  # 4 - 1 = 3
        self.assertEqual(delete_result["changes"]["deleted_rows"], 1)
        
        # 6. Check version history
        updated_info = self.dataset_service.get_dataset_info(self.user_id, self.filename)
        self.assertEqual(updated_info["total_versions"], 3)  # original + add + delete
        
        # 7. Get version diff
        versions = updated_info["versions"]
        original_version = versions[0]["version_id"]
        current_version = versions[-1]["version_id"]
        
        diff = self.dataset_service.get_version_diff(
            self.user_id, self.filename, original_version, current_version
        )
        self.assertEqual(diff["row_difference"], 0)  # 3 -> 4 -> 3 = no change
        
        # 8. Revert to original
        revert_result = self.dataset_service.revert_to_version(
            self.user_id, self.filename, original_version, "Reverted to original"
        )
        self.assertEqual(revert_result["row_count"], 3)
        self.assertEqual(revert_result["operation"], "revert")
        
        # 9. Verify final state
        final_info = self.dataset_service.get_dataset_info(self.user_id, self.filename)
        self.assertEqual(final_info["total_versions"], 4)  # original + add + delete + revert
        self.assertEqual(final_info["row_count"], 3)
    
    def test_cross_service_integration(self):
        """Test integration between user and dataset services"""
        # Create user and dataset
        self.user_service.create_or_get_user(self.user_id)
        self.user_service.add_user_file(self.user_id, self.filename, 1024)
        self.dataset_service.create_dataset_from_dataframe(self.user_id, self.filename, self.test_df)
        
        # Update file analysis count
        self.user_service.update_file_analysis(self.user_id, self.filename)
        
        # Check user stats reflect the analysis
        stats = self.user_service.get_user_stats(self.user_id)
        self.assertEqual(stats["total_analyses"], 1)
        
        # Check file analysis count
        files = self.user_service.get_user_files(self.user_id)
        self.assertEqual(files[0].analysis_count, 1)
        self.assertIsNotNone(files[0].last_analysis)
    
    def test_repository_independence(self):
        """Test that application layer is independent of infrastructure"""
        # This test verifies that we can swap repositories without changing application logic
        
        # Create a mock repository (in real tests, you'd use proper mocks)
        class MockUserRepository:
            def __init__(self):
                self.users = {}
                self.files = {}
            
            def get_by_id(self, user_id):
                return self.users.get(user_id)
            
            def save(self, user):
                self.users[user.user_id] = user
            
            def delete(self, user_id):
                return self.users.pop(user_id, None) is not None
            
            def list_all(self):
                return list(self.users.keys())
            
            def get_user_files(self, user_id):
                return self.files.get(user_id, [])
            
            def add_user_file(self, user_id, user_file):
                if user_id not in self.files:
                    self.files[user_id] = []
                self.files[user_id].append(user_file)
            
            def remove_user_file(self, user_id, filename):
                if user_id in self.files:
                    self.files[user_id] = [f for f in self.files[user_id] if f.filename != filename]
                    return True
                return False
            
            def update_file_analysis(self, user_id, filename):
                pass  # Mock implementation
        
        # Use mock repository with same application service
        mock_repository = MockUserRepository()
        mock_user_service = UserService(mock_repository)
        
        # Test that application logic works the same
        user = mock_user_service.create_or_get_user("mock_user")
        self.assertEqual(user.user_id, "mock_user")
        
        # This demonstrates that the application layer is decoupled from infrastructure


if __name__ == '__main__':
    unittest.main()