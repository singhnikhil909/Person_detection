#!/usr/bin/env python3
"""
Test script for the person detection system.
This script tests the detection API with sample images.
"""

import requests
import json
import base64
from pathlib import Path
import sys

def test_api_status(base_url="http://localhost:5000"):
    """Test API status endpoint."""
    print("Testing API status...")
    try:
        response = requests.get(f"{base_url}/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ API Status:")
            print(f"   Status: {data['status']}")
            print(f"   Embeddings loaded: {data['embeddings_loaded']}")
            print(f"   Total persons: {data.get('total_persons', 'N/A')}")
            print(f"   Total face encodings: {data.get('total_face_encodings', 'N/A')}")
            return True
        else:
            print(f"❌ API status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking API status: {str(e)}")
        return False

def test_image_upload(base_url="http://localhost:5000", image_path=None):
    """Test image upload detection."""
    if not image_path or not Path(image_path).exists():
        print("❌ No valid image path provided for upload test")
        return False
    
    print(f"Testing image upload with: {image_path}")
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'tolerance': '0.6'}
            response = requests.post(f"{base_url}/detect", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Image upload detection successful:")
            print(f"   Persons found: {result['persons_found']}")
            print(f"   Message: {result['message']}")
            
            for i, detection in enumerate(result['detections']):
                print(f"   Person {i+1}: {detection['person_name']} (confidence: {detection['confidence']:.3f})")
            
            return True
        else:
            print(f"❌ Image upload detection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error in image upload test: {str(e)}")
        return False

def test_base64_detection(base_url="http://localhost:5000", image_path=None):
    """Test base64 image detection."""
    if not image_path or not Path(image_path).exists():
        print("❌ No valid image path provided for base64 test")
        return False
    
    print(f"Testing base64 detection with: {image_path}")
    try:
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Send request
        data = {
            'image_data': image_data,
            'tolerance': 0.6
        }
        response = requests.post(f"{base_url}/detect_base64", json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Base64 detection successful:")
            print(f"   Persons found: {result['persons_found']}")
            print(f"   Message: {result['message']}")
            
            for i, detection in enumerate(result['detections']):
                print(f"   Person {i+1}: {detection['person_name']} (confidence: {detection['confidence']:.3f})")
            
            return True
        else:
            print(f"❌ Base64 detection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error in base64 test: {str(e)}")
        return False

def main():
    """Main test function."""
    print("Person Detection System - API Test")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test 1: API Status
    print("\n1. Testing API Status")
    print("-" * 30)
    if not test_api_status(base_url):
        print("❌ API is not running or not accessible")
        print("Please start the server with: python deploy_detector.py")
        return 1
    
    # Test 2: Find a test image
    print("\n2. Looking for test images")
    print("-" * 30)
    
    # Look for images in data folder
    data_folder = Path("data")
    test_image = None
    
    if data_folder.exists():
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        for ext in image_extensions:
            images = list(data_folder.glob(f"*{ext}"))
            if images:
                test_image = str(images[0])
                break
    
    if not test_image:
        print("❌ No test images found in data folder")
        print("Please ensure there are images in the data folder")
        return 1
    
    print(f"✅ Found test image: {test_image}")
    
    # Test 3: Image Upload Detection
    print("\n3. Testing Image Upload Detection")
    print("-" * 30)
    test_image_upload(base_url, test_image)
    
    # Test 4: Base64 Detection
    print("\n4. Testing Base64 Detection")
    print("-" * 30)
    test_base64_detection(base_url, test_image)
    
    print("\n" + "=" * 50)
    print("✅ API testing complete!")
    print("=" * 50)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
