#!/usr/bin/env python3
"""
Script to generate embeddings from the data folder.
Run this script once to create embeddings that can be used by the detection system.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from embedding_generator import PersonEmbeddingGenerator

def main():
    """
    Main function to generate embeddings from the data folder.
    """
    print("Person Detection System - Embedding Generator")
    print("=" * 50)
    
    # Check if data folder exists
    data_folder = Path("data")
    if not data_folder.exists():
        print(f"Error: Data folder '{data_folder}' not found!")
        print("Please ensure the data folder contains person images.")
        return 1
    
    # Count images in data folder
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_folder.glob(f"*{ext}"))
        image_files.extend(data_folder.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} image files in data folder")
    
    if len(image_files) == 0:
        print("No image files found in data folder!")
        return 1
    
    # Initialize generator
    print("\nInitializing embedding generator...")
    generator = PersonEmbeddingGenerator()
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    try:
        processed, skipped = generator.generate_embeddings()
        
        print(f"\n" + "=" * 50)
        print("EMBEDDING GENERATION COMPLETE!")
        print("=" * 50)
        print(f"✅ Processed: {processed} images")
        print(f"⚠️  Skipped: {skipped} images")
        print(f"📁 Embeddings saved to: {generator.output_folder}")
        
        # Show summary
        summary_file = generator.output_folder / "embedding_summary.json"
        if summary_file.exists():
            import json
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            print(f"👥 Total persons: {summary['total_persons']}")
            print(f"🎭 Total face embeddings: {summary['total_embeddings']}")
        
        print("\n🚀 You can now deploy the detection system!")
        print("   Run: python person_detector.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error generating embeddings: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
