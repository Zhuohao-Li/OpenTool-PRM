#!/usr/bin/env python3
"""
Simple test script for google_search function
"""

import os
import sys
import json

# Import rlab_api directly
from utils.rlab_api import rlab_api

def google_search(query, method, request_id='default', agent_version=''):
    """
    使用Google搜索API
    """
    data = {'payload': {'query': query, "method": method}}
    try:
        response = rlab_api.request(
            inst_id="ICBU_P0715_2610228AC",
            request_id=request_id,
            data=data,
            agent_version=agent_version,
        )
        
        # 检查响应是否有效
        if not response:
            return {"error": "Empty response from server"}
        
        if "payload" not in response:
            return {"error": "Invalid response format", "response": response}
        
        if "result" not in response["payload"]:
            return {"error": "No result in payload", "payload": response["payload"]}
        
        print("DEBUG: raw search_result =", repr(response["payload"]["result"]))
        return response["payload"]["result"]
        
    except Exception as e:
        return {"error": f"Network or API error: {str(e)}"}

def main():
    """Test the google_search function"""
    try:
        print("Testing google_search function...")
        result = google_search(["red dress", "tesla"], "google")
        print("Search result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 检查是否有错误
        if isinstance(result, dict) and "error" in result:
            print(f"\n⚠️  Error occurred: {result['error']}")
            print("This is likely because:")
            print("1. The RLab service is not accessible from your network")
            print("2. You need to be connected to Alibaba's internal network")
            print("3. The service might be down or the endpoint has changed")
        else:
            print("\n✅ Search completed successfully!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 

# import requests
# import json

# url = "https://google.serper.dev/search"

# payload = json.dumps({
#   "q": "apple inc"
# })
# headers = {
#   'X-API-KEY': 'fd02112c9de6543029cca9210271fadb2f191697',
#   'Content-Type': 'application/json'
# }

# response = requests.request("POST", url, headers=headers, data=payload)

# print(response.text)