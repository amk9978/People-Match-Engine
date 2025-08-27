#!/usr/bin/env python3

import pandas as pd
from services.dataset_service import dataset_service


def create_test_dataset():
    """Create a test CSV file"""
    data = {
        "Person Name": ["Alice Johnson", "Bob Smith", "Carol Davis", "David Wilson", "Eve Brown"],
        "Person Title": ["Software Engineer", "Product Manager", "Data Scientist", "CTO", "Designer"],
        "Person Company": ["TechCorp", "InnovateLabs", "DataCorp", "StartupX", "CreativeStudio"],
        "Professional Identity - Role Specification": ["Backend Development", "Product Strategy", "Machine Learning", "Technical Leadership", "UI/UX Design"],
        "Professional Identity - Experience Level": ["Mid-level", "Senior", "Mid-level", "Executive", "Junior"],
        "Company Identity - Industry Classification": ["Technology", "Technology", "Analytics", "Startup", "Design"],
        "Company Market - Market Traction": ["Growth", "Established", "Growth", "Early-stage", "Boutique"],
        "Company Offering - Value Proposition": ["SaaS Platform", "Mobile Apps", "Data Analytics", "AI Solutions", "Design Services"],
        "All Persona Titles": ["Developer;Engineer", "Manager;Strategist", "Analyst;Scientist", "Leader;Executive", "Designer;Creative"]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("test_dataset.csv", index=False)
    return df


def test_dataset_service():
    """Test dataset modification functionality"""
    print("Testing Dataset Modification Service...")
    
    # Create test dataset
    print("\n1. Creating test dataset...")
    test_df = create_test_dataset()
    print(f"Created dataset with {len(test_df)} rows")
    
    user_id = "test_user_dataset"
    filename = "test_dataset.csv"
    
    # Store original dataset
    print("\n2. Storing original dataset...")
    result = dataset_service.store_original_dataset(user_id, filename, test_df)
    print(f"Stored original: {result}")
    
    # Get dataset info
    print("\n3. Getting dataset info...")
    info = dataset_service.get_dataset_info(user_id, filename)
    print(f"Dataset info: {info}")
    
    # Preview dataset
    print("\n4. Getting dataset preview...")
    preview = dataset_service.get_dataset_preview(user_id, filename, limit=3)
    print(f"Preview: {preview}")
    
    # Add new rows
    print("\n5. Adding new rows...")
    new_rows = [
        {
            "Person Name": "Frank Miller",
            "Person Title": "DevOps Engineer", 
            "Person Company": "CloudCorp",
            "Professional Identity - Role Specification": "Infrastructure",
            "Professional Identity - Experience Level": "Senior",
            "Company Identity - Industry Classification": "Cloud Services",
            "Company Market - Market Traction": "Growth",
            "Company Offering - Value Proposition": "Cloud Infrastructure",
            "All Persona Titles": "Engineer;DevOps"
        },
        {
            "Person Name": "Grace Chen",
            "Person Title": "Marketing Manager",
            "Person Company": "BrandCo", 
            "Professional Identity - Role Specification": "Marketing Strategy",
            "Professional Identity - Experience Level": "Mid-level",
            "Company Identity - Industry Classification": "Marketing",
            "Company Market - Market Traction": "Established",
            "Company Offering - Value Proposition": "Brand Solutions",
            "All Persona Titles": "Manager;Marketer"
        }
    ]
    
    add_result = dataset_service.add_rows(user_id, filename, new_rows, "Added two new professionals")
    print(f"Add result: {add_result}")
    
    # Preview after adding
    print("\n6. Preview after adding rows...")
    preview_after_add = dataset_service.get_dataset_preview(user_id, filename, limit=8)
    print(f"New row count: {preview_after_add['total_rows']}")
    
    # Delete rows by index
    print("\n7. Deleting rows by index...")
    delete_result = dataset_service.delete_rows(user_id, filename, [0, 2], "Removed first and third person")
    print(f"Delete result: {delete_result}")
    
    # Delete rows by criteria
    print("\n8. Deleting rows by criteria...")
    criteria = {
        "Professional Identity - Experience Level": "Junior"
    }
    criteria_result = dataset_service.delete_rows_by_criteria(user_id, filename, criteria, "Removed junior level employees")
    print(f"Criteria delete result: {criteria_result}")
    
    # Get final dataset info
    print("\n9. Final dataset info...")
    final_info = dataset_service.get_dataset_info(user_id, filename)
    print(f"Final info: {final_info}")
    print(f"Total versions: {final_info['total_versions']}")
    
    # Get version diff
    print("\n10. Comparing versions...")
    if len(final_info['versions']) >= 2:
        v1 = final_info['versions'][0]['version_id']  # Original
        v2 = final_info['versions'][-1]['version_id']  # Latest
        diff = dataset_service.get_version_diff(user_id, filename, v1, v2)
        print(f"Version diff: {diff}")
    
    # Test revert
    print("\n11. Testing revert...")
    if len(final_info['versions']) >= 2:
        original_version = final_info['original_version_id']
        revert_result = dataset_service.revert_to_version(user_id, filename, original_version, "Reverted to original")
        print(f"Revert result: {revert_result}")
    
    print("\n✅ Dataset modification testing complete!")


if __name__ == "__main__":
    test_dataset_service()