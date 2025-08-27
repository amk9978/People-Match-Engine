#!/usr/bin/env python3

import requests
import json

def test_user_api():
    """Test user management API endpoints"""
    base_url = "http://localhost:8000"
    user_id = "test_api_user_456"
    headers = {"X-User-ID": user_id}
    
    print("Testing User Management API...")
    
    try:
        # Test getting current user (should create user)
        print("\n1. Getting current user...")
        response = requests.get(f"{base_url}/users/me", headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error: {response.text}")
        
        # Test getting user files
        print("\n2. Getting user files...")
        response = requests.get(f"{base_url}/users/me/files", headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error: {response.text}")
        
        # Test admin endpoint (list all users)
        print("\n3. Listing all users (admin)...")
        response = requests.get(f"{base_url}/admin/users")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error: {response.text}")
            
        # Test health check
        print("\n4. Health check...")
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ API is healthy")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running with: uvicorn api_controller_clean:app --reload")
    except Exception as e:
        print(f"❌ Error testing API: {e}")


if __name__ == "__main__":
    test_user_api()