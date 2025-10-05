#!/usr/bin/env python3
"""
Production deployment script for Person Detection System
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are met."""
    print("🔍 Checking deployment requirements...")
    
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
        print("\nPlease run 'python embedding_generator.py' first to create embeddings.")
        return False
    
    print("✅ Embedding files found")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Error: Python 3.7+ required")
        return False
    
    print("✅ Python version OK")
    
    # Check required packages
    try:
        import cv2
        import numpy as np
        import flask
        print("✅ Required packages found")
    except ImportError as e:
        print(f"❌ Error: Missing package - {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True

def start_production_server():
    """Start the production server."""
    print("\n🚀 Starting production server...")
    
    # Set production environment
    os.environ['FLASK_ENV'] = 'production'
    os.environ['FLASK_DEBUG'] = 'False'
    
    try:
        # Import and start the app
        from person_detector import app, initialize_detector
        
        # Initialize detector
        print("Initializing person detector...")
        initialize_detector()
        print("✅ Person detector initialized")
        
        # Start server
        print("\n" + "="*60)
        print("🎉 Person Detection System - Production Server")
        print("="*60)
        print("Server Status: ✅ RUNNING")
        print("Web Interface: http://0.0.0.0:5000")
        print("API Status: http://0.0.0.0:5000/status")
        print("\nFeatures:")
        print("  • Web UI for image upload and detection")
        print("  • REST API for programmatic access")
        print("  • Real-time person detection with bounding boxes")
        print("  • No cmake dependencies required!")
        print("\nPress Ctrl+C to stop the server")
        print("="*60)
        
        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        return 1

def main():
    """Main deployment function."""
    print("Person Detection System - Production Deployment")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Deployment failed - requirements not met")
        return 1
    
    # Start server
    return start_production_server()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
