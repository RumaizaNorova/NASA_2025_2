#!/usr/bin/env python3
"""
Test script to verify the production setup
"""

import os
import sys
import requests
import json
from pathlib import Path

def test_file_structure():
    """Test if all required files exist"""
    print("🔍 Testing file structure...")
    
    required_files = [
        "results_full/models/gradientboosting_model.pkl",
        "integrated_data_full.csv",
        "results_full/model_performance_full.json",
        ".env"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path} exists")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files exist")
    return True

def test_environment_variables():
    """Test if environment variables are set"""
    print("\n🔍 Testing environment variables...")
    
    # Load .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️  python-dotenv not installed, skipping .env loading")
    
    required_vars = [
        "OPENAI_API_KEY",
        "MAPBOX_PUBLIC_TOKEN"
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            # Mask the value for security
            masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"✅ {var} is set ({masked_value})")
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("💡 Make sure your .env file contains these variables")
        return False
    
    print("✅ All required environment variables are set")
    return True

def test_backend_health():
    """Test if backend is responding"""
    print("\n🔍 Testing backend health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend is healthy: {data}")
            return True
        else:
            print(f"❌ Backend returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend is not responding: {e}")
        return False

def test_frontend_access():
    """Test if frontend is accessible"""
    print("\n🔍 Testing frontend access...")
    
    try:
        response = requests.get("http://localhost:3000", timeout=10)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            return True
        else:
            print(f"❌ Frontend returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Frontend is not accessible: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("\n🔍 Testing API endpoints...")
    
    endpoints = [
        ("/", "Root endpoint"),
        ("/shark-tracks", "Shark tracks endpoint"),
        ("/species", "Species endpoint"),
        ("/stats", "Stats endpoint")
    ]
    
    all_passed = True
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=10)
            if response.status_code == 200:
                print(f"✅ {description} is working")
            else:
                print(f"❌ {description} returned status code: {response.status_code}")
                all_passed = False
        except requests.exceptions.RequestException as e:
            print(f"❌ {description} failed: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests"""
    print("🐋 Shark Habitat Prediction Dashboard - Setup Test")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_environment_variables,
        test_backend_health,
        test_frontend_access,
        test_api_endpoints
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready for production.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
