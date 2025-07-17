import requests
import json

def test_search_server():
    """Test the rlab_search.py server"""
    
    # Test 1: Basic search test
    print("=== Test 1: Basic Search ===")
    payload = {
        "queries": ["artificial intelligence machine learning"]
    }
    
    print("Sending request to: http://127.0.0.1:8000/retrieve")
    print("Payload:", json.dumps(payload, indent=2))
    
    try:
        response = requests.post("http://127.0.0.1:8000/retrieve", json=payload, timeout=30)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response structure:")
            print(f"Keys: {list(result.keys())}")
            print(f"Number of query results: {len(result.get('result', []))}")
            
            # Check the structure of the first result
            if result.get('result') and len(result['result']) > 0:
                first_result = result['result'][0]
                print(f"First query result count: {len(first_result)}")
                if len(first_result) > 0:
                    print("Sample result structure:")
                    print(json.dumps(first_result[0], indent=5, ensure_ascii=False)[:500] + "...")
                # second_result = result['result'][1]
                # print(f"Second query result count: {len(second_result)}")
                # if len(second_result) > 0:
                #     print("Sample result structure:")
                #     print(json.dumps(second_result[0], indent=5, ensure_ascii=False)[:500] + "...")
        else:
            print(f"Error response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Connection error: Cannot connect to the server.")
        print("Make sure to start the server with:")
        print("python search_r1/search/rlab_search.py --topk 5 --snippet_only")
        return False
    except requests.exceptions.Timeout:
        print("Timeout error: Request took too long to respond.")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Multiple queries test
    print("=== Test 2: Multiple Queries ===")
    payload = {
        "queries": [
            "Python programming tutorial",
            "machine learning algorithms",
            "web development frameworks"
        ]
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/retrieve", json=payload, timeout=30)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Number of queries processed: {len(result.get('result', []))}")
            print(result)
            
            for i, query_result in enumerate(result.get('result', [])):
                print(f"Query {i+1} returned {len(query_result)} results")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Error in multiple queries test: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Edge cases
    print("=== Test 3: Edge Cases ===")
    
    # Empty query test
    print("Testing empty query...")
    payload = {"queries": [""]}
    try:
        response = requests.post("http://127.0.0.1:8000/retrieve", json=payload, timeout=30)
        print(f"Empty query status: {response.status_code}")
    except Exception as e:
        print(f"Empty query error: {e}")
    
    # Special characters test
    print("Testing special characters...")
    payload = {"queries": ["C++ programming & development"]}
    try:
        response = requests.post("http://127.0.0.1:8000/retrieve", json=payload, timeout=30)
        print(f"Special chars status: {response.status_code}")
    except Exception as e:
        print(f"Special chars error: {e}")
    
    return True

def test_server_endpoints():
    """Test server endpoints and API structure"""
    print("=== Testing Server Endpoints ===")
    
    # Test root endpoint
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=10)
        print(f"Root endpoint status: {response.status_code}")
    except Exception as e:
        print(f"Root endpoint error: {e}")
    
    # Test docs endpoint
    try:
        response = requests.get("http://127.0.0.1:8000/docs", timeout=10)
        print(f"Docs endpoint status: {response.status_code}")
    except Exception as e:
        print(f"Docs endpoint error: {e}")

if __name__ == "__main__":
    print("Starting rlab_search.py server tests...")
    print("Make sure the server is running with:")
    print("python search_r1/search/rlab_search.py --topk 3 --snippet_only")
    print("\n" + "="*50 + "\n")
    
    success = test_search_server()
    
    if success:
        test_server_endpoints()
        print("\n=== Test Summary ===")
        print("✓ Basic functionality test completed")
        print("✓ Multiple queries test completed") 
        print("✓ Edge cases test completed")
        print("✓ Server endpoints test completed")
        print("\nIf you see search results above, the server is working correctly!")
    else:
        print("\n=== Test Failed ===")
        print("Please check:")
        print("1. Server is running on port 8000")
        print("2. utils.rlab_api module is available")
        print("3. rlab API credentials are configured")