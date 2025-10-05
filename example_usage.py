#!/usr/bin/env python3
"""
Example usage of the Person Detection System.
This script demonstrates how to use the detection system programmatically.
"""

import sys
from pathlib import Path
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from person_detector import PersonDetector
import cv2
import numpy as np

def example_detection():
    """Example of using the person detection system."""
    print("Person Detection System - Example Usage")
    print("=" * 50)
    
    # Initialize detector
    print("Initializing person detector...")
    try:
        detector = PersonDetector()
        print("✅ Detector initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing detector: {str(e)}")
        print("Please run 'python generate_embeddings.py' first")
        return 1
    
    # Find a test image
    print("\nLooking for test images...")
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
        return 1
    
    print(f"✅ Found test image: {test_image}")
    
    # Detect persons in the image
    print(f"\nDetecting persons in: {test_image}")
    print("-" * 40)
    
    result = detector.detect_persons_in_image(test_image, tolerance=0.6)
    
    if result['success']:
        print(f"✅ Detection successful!")
        print(f"   Persons found: {result['persons_found']}")
        print(f"   Message: {result['message']}")
        
        if result['detections']:
            print("\nDetected persons:")
            for i, detection in enumerate(result['detections']):
                print(f"   Person {i+1}:")
                print(f"     Name: {detection['person_name']}")
                print(f"     Confidence: {detection['confidence']:.3f}")
                print(f"     Face location: {detection['face_location']}")
        else:
            print("   No persons detected")
    else:
        print(f"❌ Detection failed: {result['message']}")
        return 1
    
    # Example of drawing detections
    print(f"\nDrawing detections on image...")
    output_path = "detection_result.jpg"
    
    if detector.draw_detections(test_image, output_path, tolerance=0.6):
        print(f"✅ Detection result saved to: {output_path}")
    else:
        print("❌ Failed to draw detections")
    
    return 0

def example_api_usage():
    """Example of using the API endpoints."""
    print("\n" + "=" * 50)
    print("API Usage Examples")
    print("=" * 50)
    
    print("\n1. Check API status:")
    print("   curl http://localhost:5000/status")
    
    print("\n2. Upload image for detection:")
    print("   curl -X POST -F \"image=@test_image.jpg\" -F \"tolerance=0.6\" http://localhost:5000/detect")
    
    print("\n3. Send base64 image:")
    print("   curl -X POST -H \"Content-Type: application/json\" \\")
    print("     -d '{\"image_data\": \"base64_encoded_image\", \"tolerance\": 0.6}' \\")
    print("     http://localhost:5000/detect_base64")
    
    print("\n4. Python requests example:")
    print("""
import requests

# Upload image
with open('test_image.jpg', 'rb') as f:
    files = {'image': f}
    data = {'tolerance': '0.6'}
    response = requests.post('http://localhost:5000/detect', files=files, data=data)
    result = response.json()
    print(result)
""")

def main():
    """Main function."""
    print("Person Detection System - Example Usage")
    print("=" * 50)
    
    # Check if embeddings exist
    embeddings_folder = Path("embeddings")
    required_files = [
        "person_embeddings.pkl",
        "face_locations.pkl", 
        "person_metadata.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not (embeddings_folder / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Embedding files not found!")
        print(f"Missing files: {', '.join(missing_files)}")
        print("\nPlease run 'python generate_embeddings.py' first to create embeddings.")
        return 1
    
    # Run example detection
    result = example_detection()
    
    # Show API usage examples
    example_api_usage()
    
    return result

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
