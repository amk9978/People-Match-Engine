#!/usr/bin/env python3
"""
Dataset Modification API Examples

This file shows how to use the dataset modification API endpoints 
to add/delete rows from uploaded CSV files and analyze modified versions.
"""

import requests
import json


BASE_URL = "http://localhost:8000"
USER_ID = "example_user_123"
HEADERS = {"X-User-ID": USER_ID}


def example_1_list_datasets():
    """Example 1: List all datasets for user"""
    print("=== Example 1: List User Datasets ===")
    
    response = requests.get(f"{BASE_URL}/datasets", headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {data['total_datasets']} datasets:")
        for dataset in data['datasets']:
            print(f"  • {dataset['filename']} ({dataset['current_rows']} rows, {dataset['total_versions']} versions)")
    else:
        print(f"❌ Error: {response.text}")


def example_2_get_dataset_info(filename):
    """Example 2: Get detailed dataset information"""
    print(f"\n=== Example 2: Get Dataset Info for '{filename}' ===")
    
    response = requests.get(f"{BASE_URL}/datasets/{filename}", headers=HEADERS)
    
    if response.status_code == 200:
        info = response.json()
        print(f"✅ Dataset Info:")
        print(f"  • Current rows: {info['row_count']}")
        print(f"  • Total versions: {info['total_versions']}")
        print(f"  • Columns: {len(info['columns'])}")
        print(f"  • Version history:")
        for v in info['versions'][-3:]:  # Show last 3 versions
            changes = v['changes']
            print(f"    - {v['version_id'][:8]}: {v['type']} (+{changes['added_rows']}/-{changes['deleted_rows']} rows) - {v['description']}")
    else:
        print(f"❌ Error: {response.text}")


def example_3_preview_dataset(filename):
    """Example 3: Preview dataset rows"""
    print(f"\n=== Example 3: Preview Dataset '{filename}' ===")
    
    response = requests.get(f"{BASE_URL}/datasets/{filename}/preview?limit=3", headers=HEADERS)
    
    if response.status_code == 200:
        preview = response.json()
        print(f"✅ Preview ({preview['preview_rows']} of {preview['total_rows']} rows):")
        for i, row in enumerate(preview['data'], 1):
            name = row.get('Person Name', 'Unknown')
            title = row.get('Person Title', 'Unknown')
            company = row.get('Person Company', 'Unknown')
            print(f"  {i}. {name} - {title} at {company}")
    else:
        print(f"❌ Error: {response.text}")


def example_4_add_rows(filename):
    """Example 4: Add new rows to dataset"""
    print(f"\n=== Example 4: Add Rows to '{filename}' ===")
    
    new_rows_data = {
        "rows": [
            {
                "Person Name": "John Doe",
                "Person Title": "Senior Developer",
                "Person Company": "TechStart",
                "Professional Identity - Role Specification": "Full-Stack Development",
                "Professional Identity - Experience Level": "Senior",
                "Company Identity - Industry Classification": "Technology",
                "Company Market - Market Traction": "Growth",
                "Company Offering - Value Proposition": "Web Applications",
                "All Persona Titles": "Developer;Engineer;Programmer"
            },
            {
                "Person Name": "Sarah Wilson",
                "Person Title": "UX Designer",
                "Person Company": "DesignHub",
                "Professional Identity - Role Specification": "User Experience Design",
                "Professional Identity - Experience Level": "Mid-level",
                "Company Identity - Industry Classification": "Design",
                "Company Market - Market Traction": "Established",
                "Company Offering - Value Proposition": "Design Services",
                "All Persona Titles": "Designer;Creative;UX"
            }
        ],
        "description": "Added two new professionals from different domains"
    }
    
    response = requests.post(
        f"{BASE_URL}/datasets/{filename}/add-rows",
        headers={**HEADERS, "Content-Type": "application/json"},
        json=new_rows_data
    )
    
    if response.status_code == 200:
        result = response.json()
        version_info = result['result']
        print(f"✅ {result['message']}")
        print(f"  • New version: {version_info['version_id']}")
        print(f"  • Total rows: {version_info['row_count']}")
        print(f"  • Added: {version_info['changes']['added_rows']} rows")
        return version_info['version_id']
    else:
        print(f"❌ Error: {response.text}")
        return None


def example_5_delete_by_criteria(filename):
    """Example 5: Delete rows matching criteria"""
    print(f"\n=== Example 5: Delete Rows by Criteria from '{filename}' ===")
    
    # Delete all rows where experience level is "Junior"
    delete_criteria = {
        "criteria": {
            "Professional Identity - Experience Level": "Junior"
        },
        "description": "Removed all junior-level positions"
    }
    
    response = requests.post(
        f"{BASE_URL}/datasets/{filename}/delete-rows-by-criteria",
        headers={**HEADERS, "Content-Type": "application/json"},
        json=delete_criteria
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ {result['message']}")
        return result['result']['version_id']
    else:
        print(f"❌ Error: {response.text}")
        return None


def example_6_compare_versions(filename, version1, version2):
    """Example 6: Compare two dataset versions"""
    print(f"\n=== Example 6: Compare Versions of '{filename}' ===")
    
    response = requests.get(
        f"{BASE_URL}/datasets/{filename}/diff?version1={version1}&version2={version2}",
        headers=HEADERS
    )
    
    if response.status_code == 200:
        diff = response.json()
        print(f"✅ Version Comparison:")
        print(f"  • {diff['summary']}")
        print(f"  • Version {version1}: {diff['version1']['rows']} rows")
        print(f"  • Version {version2}: {diff['version2']['rows']} rows")
        if diff['columns_changed']:
            print(f"  • Column changes: {diff['columns_changed']}")
    else:
        print(f"❌ Error: {response.text}")


def example_7_revert_version(filename, target_version):
    """Example 7: Revert to a previous version"""
    print(f"\n=== Example 7: Revert '{filename}' to Version {target_version} ===")
    
    response = requests.post(
        f"{BASE_URL}/datasets/{filename}/revert/{target_version}?description=Reverted%20to%20original%20state",
        headers=HEADERS
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ {result['message']}")
        print(f"  • New version: {result['result']['version_id']}")
        return result['result']['version_id']
    else:
        print(f"❌ Error: {response.text}")
        return None


def example_8_analyze_modified(filename, version_id=None):
    """Example 8: Analyze modified dataset"""
    print(f"\n=== Example 8: Analyze Modified Dataset '{filename}' ===")
    
    params = {"min_density": 0.2}
    if version_id:
        params["version_id"] = version_id
    
    response = requests.post(
        f"{BASE_URL}/analyze-modified/{filename}",
        headers=HEADERS,
        params=params
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Analysis started:")
        print(f"  • Job ID: {result['job_id']}")
        print(f"  • Status: {result['status']}")
        print(f"  • Message: {result['message']}")
        return result['job_id']
    else:
        print(f"❌ Error: {response.text}")
        return None


def run_complete_example():
    """Run a complete example workflow"""
    print("🚀 Running Complete Dataset Modification Workflow")
    print("=" * 60)
    
    filename = "example_dataset.csv"
    
    # Note: You need to upload a CSV via POST /analyze first to have a dataset
    print("📋 Prerequisites:")
    print("1. Start the API server: uvicorn api_controller_clean:app --reload")
    print("2. Upload a CSV file via POST /analyze with X-User-ID header")
    print(f"3. Use filename '{filename}' in the examples below")
    print()
    
    try:
        # Run examples
        example_1_list_datasets()
        example_2_get_dataset_info(filename)
        example_3_preview_dataset(filename)
        
        # Modify dataset
        new_version = example_4_add_rows(filename)
        if new_version:
            example_5_delete_by_criteria(filename)
            
            # Get updated info to see versions
            example_2_get_dataset_info(filename)
            
            # Compare versions and revert
            response = requests.get(f"{BASE_URL}/datasets/{filename}", headers=HEADERS)
            if response.status_code == 200:
                info = response.json()
                versions = info['versions']
                if len(versions) >= 2:
                    original = versions[0]['version_id']
                    current = versions[-1]['version_id']
                    
                    example_6_compare_versions(filename, original, current)
                    reverted_version = example_7_revert_version(filename, original)
                    
                    if reverted_version:
                        example_8_analyze_modified(filename, reverted_version)
        
        print("\n✅ Complete workflow demonstration finished!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API server.")
        print("Start the server with: uvicorn api_controller_clean:app --reload")


if __name__ == "__main__":
    run_complete_example()