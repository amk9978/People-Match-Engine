#!/usr/bin/env python3

from services.user_service import user_service


def test_user_service():
    """Test user service functionality"""
    print("Testing User Service...")
    
    # Test user creation
    user_id = "test_user_123"
    print(f"\n1. Creating user: {user_id}")
    user_data = user_service.create_user(user_id)
    print(f"Created user: {user_data}")
    
    # Test adding files
    print("\n2. Adding files...")
    file1 = user_service.add_user_file(user_id, "test_batch1.csv", 1024)
    print(f"Added file 1: {file1}")
    
    file2 = user_service.add_user_file(user_id, "test_batch2.csv", 2048)
    print(f"Added file 2: {file2}")
    
    # Test getting user files
    print("\n3. Getting user files...")
    files = user_service.get_user_files(user_id)
    print(f"User files: {files}")
    
    # Test updating analysis count
    print("\n4. Updating analysis counts...")
    user_service.update_file_analysis(user_id, "test_batch1.csv")
    user_service.update_file_analysis(user_id, "test_batch1.csv")
    user_service.update_file_analysis(user_id, "test_batch2.csv")
    
    # Test user stats
    print("\n5. Getting user stats...")
    stats = user_service.get_user_stats(user_id)
    print(f"User stats: {stats}")
    
    # Test removing file
    print("\n6. Removing file...")
    success = user_service.remove_user_file(user_id, "test_batch1.csv")
    print(f"File removal success: {success}")
    
    # Final stats
    print("\n7. Final stats...")
    final_stats = user_service.get_user_stats(user_id)
    print(f"Final stats: {final_stats}")
    
    # Test listing all users
    print("\n8. Listing all users...")
    all_users = user_service.list_all_users()
    print(f"All users: {all_users}")


if __name__ == "__main__":
    test_user_service()