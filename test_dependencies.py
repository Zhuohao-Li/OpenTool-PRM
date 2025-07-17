#!/usr/bin/env python3
"""
Test script to verify all dependencies for rlab_search.py are available
"""

import sys
import importlib.util

def test_python_packages():
    """Test if required Python packages are installed"""
    required_packages = [
        'requests',
        'aiohttp', 
        'chardet',
        'bs4',
        'uvicorn',
        'fastapi',
        'pydantic'
    ]
    
    print("=== Testing Python Package Dependencies ===")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("All required packages are installed!")
        return True

def test_rlab_api():
    """Test if rlab_api module is available"""
    print("\n=== Testing rlab_api Module ===")
    
    try:
        # Add search_r1 to Python path
        sys.path.insert(0, 'search_r1')
        from search.utils.rlab_api import rlab_api
        print("✓ rlab_api module imported successfully")
        
        # Check if rlab_api has the required method
        if hasattr(rlab_api, 'request'):
            print("✓ rlab_api.request method is available")
            return True
        else:
            print("✗ rlab_api.request method not found")
            return False
            
    except ImportError as e:
        print(f"✗ Failed to import rlab_api: {e}")
        print("Make sure search_r1/search/utils/rlab_api.py exists and is properly configured")
        return False
    except Exception as e:
        print(f"✗ Error testing rlab_api: {e}")
        return False

def test_google_search_function():
    """Test if the google_search function can be imported"""
    print("\n=== Testing google_search Function ===")
    
    try:
        sys.path.insert(0, 'search_r1')
        from search.rlab_search import google_search
        print("✓ google_search function imported successfully")
        
        # Try a test call (this might fail due to API credentials, but should not crash)
        try:
            result = google_search("test query", "google", "test_request", "")
            print("✓ google_search function call completed")
            print(f"Result type: {type(result)}")
            return True
        except Exception as e:
            print(f"⚠ google_search function call failed: {e}")
            print("This might be due to API credentials or network issues")
            return True  # Still return True as the function exists
            
    except ImportError as e:
        print(f"✗ Failed to import google_search: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing google_search: {e}")
        return False

def main():
    """Run all dependency tests"""
    print("Starting dependency tests for rlab_search.py...")
    print("="*60)
    
    all_tests_passed = True
    
    # Test 1: Python packages
    if not test_python_packages():
        all_tests_passed = False
    
    # Test 2: rlab_api module
    if not test_rlab_api():
        all_tests_passed = False
    
    # Test 3: google_search function
    if not test_google_search_function():
        all_tests_passed = False
    
    print("\n" + "="*60)
    
    if all_tests_passed:
        print("🎉 All dependency tests PASSED!")
        print("\nYou can now test the server:")
        print("1. Start server: python search_r1/search/rlab_search.py --topk 3 --snippet_only")
        print("2. Run tests: python test_search.py")
    else:
        print("❌ Some dependency tests FAILED!")
        print("Please fix the issues above before testing the server")
    
    return all_tests_passed

if __name__ == "__main__":
    main() 