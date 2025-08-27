#!/usr/bin/env python3

import unittest
from datetime import datetime
import pandas as pd
from domain.entities.dataset import Dataset, DatasetVersion, DatasetModification, DatasetDeletionCriteria


class TestDatasetVersionEntity(unittest.TestCase):
    """Unit tests for DatasetVersion domain entity"""
    
    def test_create_original(self):
        """Test creating original version"""
        version = DatasetVersion.create_original(10, "/path/to/file.csv")
        
        self.assertEqual(version.operation_type, "original")
        self.assertEqual(version.row_count, 10)
        self.assertEqual(version.file_path, "/path/to/file.csv")
        self.assertEqual(version.changes["added_rows"], 0)
        self.assertEqual(version.changes["deleted_rows"], 0)
        self.assertEqual(version.description, "Original uploaded dataset")
    
    def test_create_modification(self):
        """Test creating modification version"""
        changes = {"added_rows": 5, "deleted_rows": 2}
        version = DatasetVersion.create_modification(
            "add_rows", 13, changes, "/path/to/modified.csv", "Added new data"
        )
        
        self.assertEqual(version.operation_type, "add_rows")
        self.assertEqual(version.row_count, 13)
        self.assertEqual(version.changes, changes)
        self.assertEqual(version.description, "Added new data")
    
    def test_to_dict(self):
        """Test dictionary conversion"""
        version = DatasetVersion.create_original(5, "/test.csv")
        result = version.to_dict()
        
        expected_keys = {
            "version_id", "type", "created_at", "row_count", 
            "changes", "file_path", "description"
        }
        self.assertEqual(set(result.keys()), expected_keys)


class TestDatasetEntity(unittest.TestCase):
    """Unit tests for Dataset domain entity"""
    
    def setUp(self):
        self.original_version = DatasetVersion.create_original(10, "/original.csv")
        self.dataset = Dataset(
            user_id="test_user",
            filename="test.csv",
            original_version_id=self.original_version.version_id,
            current_version_id=self.original_version.version_id,
            created_at=datetime(2023, 1, 1),
            column_count=5,
            columns=["col1", "col2", "col3", "col4", "col5"],
            versions=[self.original_version]
        )
    
    def test_current_row_count(self):
        """Test current row count property"""
        self.assertEqual(self.dataset.current_row_count, 10)
    
    def test_total_versions(self):
        """Test total versions property"""
        self.assertEqual(self.dataset.total_versions, 1)
    
    def test_get_version(self):
        """Test getting version by ID"""
        version = self.dataset.get_version(self.original_version.version_id)
        self.assertIsNotNone(version)
        self.assertEqual(version.version_id, self.original_version.version_id)
        
        # Test non-existent version
        self.assertIsNone(self.dataset.get_version("non_existent"))
    
    def test_add_version(self):
        """Test adding new version"""
        new_version = DatasetVersion.create_modification(
            "add_rows", 15, {"added_rows": 5, "deleted_rows": 0}, "/new.csv", "Added rows"
        )
        
        self.dataset.add_version(new_version)
        
        self.assertEqual(self.dataset.total_versions, 2)
        self.assertEqual(self.dataset.current_version_id, new_version.version_id)
        self.assertEqual(self.dataset.current_row_count, 15)
    
    def test_get_version_diff(self):
        """Test version difference calculation"""
        # Add another version
        new_version = DatasetVersion.create_modification(
            "delete_rows", 8, {"added_rows": 0, "deleted_rows": 2}, "/modified.csv", "Deleted rows"
        )
        self.dataset.add_version(new_version)
        
        diff = self.dataset.get_version_diff(
            self.original_version.version_id, 
            new_version.version_id
        )
        
        self.assertEqual(diff["row_difference"], -2)  # 8 - 10 = -2
        self.assertIn("2 rows", diff["summary"])


class TestDatasetModification(unittest.TestCase):
    """Unit tests for DatasetModification value object"""
    
    def test_validate_columns_success(self):
        """Test successful column validation"""
        modification = DatasetModification(rows=[
            {"col1": "val1", "col2": "val2"},
            {"col1": "val3", "col2": "val4"}
        ])
        
        # Should not raise exception
        modification.validate_columns(["col1", "col2"])
    
    def test_validate_columns_missing(self):
        """Test validation with missing columns"""
        modification = DatasetModification(rows=[
            {"col1": "val1"}  # Missing col2
        ])
        
        with self.assertRaises(ValueError) as cm:
            modification.validate_columns(["col1", "col2"])
        
        self.assertIn("missing columns", str(cm.exception))
    
    def test_validate_columns_extra(self):
        """Test validation with extra columns"""
        modification = DatasetModification(rows=[
            {"col1": "val1", "col2": "val2", "col3": "val3"}  # Extra col3
        ])
        
        with self.assertRaises(ValueError) as cm:
            modification.validate_columns(["col1", "col2"])
        
        self.assertIn("extra columns", str(cm.exception))


class TestDatasetDeletionCriteria(unittest.TestCase):
    """Unit tests for DatasetDeletionCriteria value object"""
    
    def setUp(self):
        self.df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "Chicago"]
        })
    
    def test_apply_simple_criteria(self):
        """Test applying simple equality criteria"""
        criteria = DatasetDeletionCriteria(criteria={"age": 30})
        
        result = criteria.apply_to_dataframe(self.df)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "Bob")
    
    def test_apply_contains_criteria(self):
        """Test applying contains operator"""
        criteria = DatasetDeletionCriteria(criteria={
            "city": {"operator": "contains", "value": "C"}
        })
        
        result = criteria.apply_to_dataframe(self.df)
        
        self.assertEqual(len(result), 1)  # Only Charlie in Chicago
        self.assertEqual(result.iloc[0]["name"], "Charlie")
    
    def test_invalid_column(self):
        """Test validation with invalid column"""
        criteria = DatasetDeletionCriteria(criteria={"invalid_col": "value"})
        
        with self.assertRaises(ValueError) as cm:
            criteria.apply_to_dataframe(self.df)
        
        self.assertIn("not found in dataset", str(cm.exception))


if __name__ == '__main__':
    unittest.main()