"""
Test NASA data access with the provided token
"""

import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_nasa_access():
    """Test what we can actually access with the token"""
    print("=== TESTING NASA DATA ACCESS ===")
    
    token = os.getenv('EARTHDATA_TOKEN')
    if not token:
        print("ERROR: No EARTHDATA_TOKEN found in .env file")
        return False
    
    print(f"Using token: {token[:10]}...")
    
    headers = {'Authorization': f'Bearer {token}'}
    
    # Test 1: Try to access the CMR API to see what datasets are available
    print("\n--- Test 1: CMR API Access ---")
    cmr_url = "https://cmr.earthdata.nasa.gov/search/collections.json"
    
    try:
        response = requests.get(cmr_url, headers=headers, timeout=30)
        print(f"CMR API status: {response.status_code}")
        if response.status_code == 200:
            print("✅ CMR API accessible")
        else:
            print(f"❌ CMR API error: {response.text[:200]}")
    except Exception as e:
        print(f"❌ CMR API error: {e}")
    
    # Test 2: Search for MODIS-Aqua datasets
    print("\n--- Test 2: Search for MODIS-Aqua datasets ---")
    search_url = "https://cmr.earthdata.nasa.gov/search/collections.json"
    params = {
        'keyword': 'MODIS-Aqua',
        'page_size': 10
    }
    
    try:
        response = requests.get(search_url, params=params, headers=headers, timeout=30)
        print(f"Search status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Found {data.get('hits', 0)} collections")
            if 'feed' in data and 'entry' in data['feed']:
                print("Available MODIS-Aqua datasets:")
                for entry in data['feed']['entry'][:5]:  # Show first 5
                    print(f"  - {entry.get('title', 'Unknown')}")
                    print(f"    ID: {entry.get('concept-id', 'Unknown')}")
        else:
            print(f"❌ Search error: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Search error: {e}")
    
    # Test 3: Try to access OceanColor data
    print("\n--- Test 3: OceanColor API Access ---")
    ocean_url = "https://oceandata.sci.gsfc.nasa.gov/api/file_search"
    
    try:
        response = requests.get(ocean_url, headers=headers, timeout=30)
        print(f"OceanColor API status: {response.status_code}")
        if response.status_code == 200:
            print("✅ OceanColor API accessible")
        else:
            print(f"❌ OceanColor API error: {response.text[:200]}")
    except Exception as e:
        print(f"❌ OceanColor API error: {e}")
    
    # Test 4: Try a simple file download test
    print("\n--- Test 4: File Download Test ---")
    # Try a small, recent file
    test_url = "https://oceandata.sci.gsfc.nasa.gov/cmr/getfile/AQUA_MODIS.20200201_20200201.L3b.MC.SST.nc"
    
    try:
        response = requests.get(test_url, headers=headers, timeout=60)
        print(f"Download test status: {response.status_code}")
        if response.status_code == 200:
            print(f"✅ File accessible, size: {len(response.content)} bytes")
        else:
            print(f"❌ Download error: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Download error: {e}")
    
    return True

if __name__ == "__main__":
    test_nasa_access()
