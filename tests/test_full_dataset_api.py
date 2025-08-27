#!/usr/bin/env python3

import requests
import json


def test_dataset_api_endpoints():
    """Test all dataset modification API endpoints"""
    base_url = "http://localhost:8000"
    user_id = "api_test_user_123"
    headers = {"X-User-ID": user_id}
    
    print("Testing Dataset Modification API Endpoints...")
    
    try:
        # 1. List datasets (should be empty initially)
        print("\n1. Listing user datasets...")
        response = requests.get(f"{base_url}/datasets", headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Found {data['total_datasets']} datasets")
        
        # Note: We need to upload a CSV first via /analyze endpoint to have a dataset
        # For this test, we'll assume a dataset exists from previous uploads
        
        # 2. Get dataset info
        filename = "test_batch2.csv"  # Assume this exists
        print(f"\n2. Getting dataset info for {filename}...")
        response = requests.get(f"{base_url}/datasets/{filename}", headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            dataset_info = response.json()
            print(f"Dataset has {dataset_info.get('row_count', 'unknown')} rows")
            print(f"Total versions: {dataset_info.get('total_versions', 0)}")
        elif response.status_code == 404:
            print("Dataset not found - need to upload via /analyze first")
            return
        
        # 3. Get dataset preview  
        print(f"\n3. Getting dataset preview...")
        response = requests.get(f"{base_url}/datasets/{filename}/preview?limit=3", headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            preview = response.json()
            print(f"Preview shows {preview['preview_rows']} of {preview['total_rows']} rows")
            print(f"Columns: {preview['columns'][:3]}...")  # Show first 3 columns
        
        # 4. Add new rows
        print(f"\n4. Adding new rows...")
        new_rows_data = {
            "rows": [
                {
                    "Person Name": "API Test User",
                    "Person Title": "Test Engineer", 
                    "Person Company": "TestCorp",
                    "Professional Identity - Role Specification": "API Testing",
                    "Professional Identity - Experience Level": "Senior",
                    "Company Identity - Industry Classification": "Technology",
                    "Company Market - Market Traction": "Growth",
                    "Company Offering - Value Proposition": "Testing Solutions",
                    "All Persona Titles": "Engineer;Tester"
                }
            ],
            "description": "Added via API test"
        }
        
        response = requests.post(
            f"{base_url}/datasets/{filename}/add-rows",
            headers={**headers, "Content-Type": "application/json"},
            json=new_rows_data
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Successfully added rows. New version: {result['result']['version_id']}")
        else:
            print(f"Error: {response.text}")
        
        # 5. Delete rows by criteria
        print(f"\n5. Deleting rows by criteria...")
        delete_criteria = {
            "criteria": {
                "Person Name": "API Test User"
            },
            "description": "Removed test user via API"
        }
        
        response = requests.post(
            f"{base_url}/datasets/{filename}/delete-rows-by-criteria",
            headers={**headers, "Content-Type": "application/json"},
            json=delete_criteria
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Successfully deleted rows. Message: {result['message']}")
        else:
            print(f"Error: {response.text}")
        
        # 6. Get updated dataset info
        print(f"\n6. Getting updated dataset info...")
        response = requests.get(f"{base_url}/datasets/{filename}", headers=headers)
        if response.status_code == 200:
            updated_info = response.json()
            print(f"Current row count: {updated_info.get('row_count')}")
            print(f"Total versions: {updated_info.get('total_versions')}")
            
            # 7. Get version diff if we have multiple versions
            versions = updated_info.get('versions', [])
            if len(versions) >= 2:
                print(f"\n7. Getting version diff...")
                v1 = versions[0]['version_id']  # Original
                v2 = versions[-1]['version_id']  # Latest
                
                response = requests.get(
                    f"{base_url}/datasets/{filename}/diff?version1={v1}&version2={v2}",
                    headers=headers
                )
                if response.status_code == 200:
                    diff = response.json()
                    print(f"Version diff: {diff['summary']}")
        
        # 8. Test analyze modified dataset
        print(f"\n8. Testing analysis of modified dataset...")
        response = requests.post(
            f"{base_url}/analyze-modified/{filename}",
            headers=headers,
            params={"min_density": 0.3}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            analysis_result = response.json()
            print(f"Analysis started. Job ID: {analysis_result['job_id']}")
        else:
            print(f"Error: {response.text}")
        
        print("\n✅ Dataset API testing complete!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running with:")
        print("python -c 'import uvicorn; from api_controller_clean import app; uvicorn.run(app, host=\"0.0.0.0\", port=8000)'")
    except Exception as e:
        print(f"❌ Error testing API: {e}")


def print_api_documentation():
    """Print API endpoint documentation"""
    print("\n" + "="*70)
    print("DATASET MODIFICATION API ENDPOINTS")
    print("="*70)
    
    endpoints = [
        ("GET", "/datasets", "List all datasets for user"),
        ("GET", "/datasets/{filename}", "Get dataset info and version history"),
        ("GET", "/datasets/{filename}/preview", "Preview dataset rows"),
        ("POST", "/datasets/{filename}/add-rows", "Add new rows to dataset"),
        ("POST", "/datasets/{filename}/delete-rows", "Delete rows by index"),
        ("POST", "/datasets/{filename}/delete-rows-by-criteria", "Delete rows matching criteria"),
        ("POST", "/datasets/{filename}/revert/{version_id}", "Revert to specific version"),
        ("GET", "/datasets/{filename}/diff", "Compare two versions"),
        ("POST", "/analyze-modified/{filename}", "Analyze modified dataset"),
    ]
    
    for method, endpoint, description in endpoints:
        print(f"{method:6} {endpoint:50} - {description}")
    
    print("\n" + "="*70)
    print("REQUIRED HEADER: X-User-ID")
    print("="*70)


if __name__ == "__main__":
    print_api_documentation()
    test_dataset_api_endpoints()