#!/usr/bin/env python3

import unittest
from datetime import datetime
from domain.entities.user import User, UserFile


class TestUserEntity(unittest.TestCase):
    """Unit tests for User domain entity"""
    
    def setUp(self):
        self.user = User(
            user_id="test_user",
            created_at=datetime(2023, 1, 1),
            last_active=datetime(2023, 1, 1),
            total_files=0,
            total_analyses=0
        )
    
    def test_update_activity(self):
        """Test user activity update"""
        original_time = self.user.last_active
        self.user.update_activity()
        self.assertGreater(self.user.last_active, original_time)
    
    def test_increment_file_count(self):
        """Test file count increment"""
        self.assertEqual(self.user.total_files, 0)
        self.user.increment_file_count()
        self.assertEqual(self.user.total_files, 1)
    
    def test_decrement_file_count(self):
        """Test file count decrement"""
        self.user.total_files = 5
        self.user.decrement_file_count()
        self.assertEqual(self.user.total_files, 4)
    
    def test_decrement_file_count_at_zero(self):
        """Test file count doesn't go below zero"""
        self.assertEqual(self.user.total_files, 0)
        self.user.decrement_file_count()
        self.assertEqual(self.user.total_files, 0)
    
    def test_increment_analysis_count(self):
        """Test analysis count increment"""
        self.assertEqual(self.user.total_analyses, 0)
        self.user.increment_analysis_count()
        self.assertEqual(self.user.total_analyses, 1)


class TestUserFileEntity(unittest.TestCase):
    """Unit tests for UserFile domain entity"""
    
    def setUp(self):
        self.user_file = UserFile(
            filename="test.csv",
            uploaded_at=datetime(2023, 1, 1),
            file_size=1024
        )
    
    def test_increment_analysis_count(self):
        """Test analysis count increment"""
        self.assertEqual(self.user_file.analysis_count, 0)
        self.assertIsNone(self.user_file.last_analysis)
        
        self.user_file.increment_analysis_count()
        
        self.assertEqual(self.user_file.analysis_count, 1)
        self.assertIsNotNone(self.user_file.last_analysis)
    
    def test_to_dict(self):
        """Test dictionary conversion"""
        result = self.user_file.to_dict()
        
        expected_keys = {
            "filename", "uploaded_at", "file_size", 
            "analysis_count", "last_analysis"
        }
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertEqual(result["filename"], "test.csv")
        self.assertEqual(result["file_size"], 1024)


if __name__ == '__main__':
    unittest.main()