#!/usr/bin/env python3
"""
Script to deploy the person detection system.
This script starts the Flask API server for person detection.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from person_detector import app, initialize_detector

def main():
    """
    Main function to deploy the detection system.
    """
    print("Person Detection System - Server Deployment")
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
        print("❌ Error: Embedding files not found!")
        print(f"Missing files: {', '.join(missing_files)}")
        print("\nPlease run 'python generate_embeddings.py' first to create embeddings.")
        return 1
    
    print("✅ Embedding files found")
    
    # Initialize detector
    print("\nInitializing person detector...")
    try:
        initialize_detector()
        print("✅ Person detector initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing detector: {str(e)}")
        return 1
    
    # Start server
    print("\n🚀 Starting Flask server...")
    print("=" * 50)
    print("Server will be available at:")
    print("  - Local: http://localhost:5000")
    print("  - Network: http://0.0.0.0:5000")
    print("\nAPI Endpoints:")
    print("  - GET  /           : API documentation")
    print("  - GET  /status     : Check system status")
    print("  - POST /detect     : Upload image for detection")
    print("  - POST /detect_base64 : Send base64 image for detection")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error running server: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
