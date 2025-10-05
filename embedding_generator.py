"""
Person Detection System - Embedding Generator Module
This module creates embeddings from person images in the data folder.
Run this once to generate embeddings that can be used by the detection module.
"""

import os
import pickle
import numpy as np
import cv2
from PIL import Image
from opencv_face_detector import OpenCVFaceDetector
from pathlib import Path
import json
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonEmbeddingGenerator:
    def __init__(self, data_folder: str = "data", output_folder: str = "embeddings"):
        """
        Initialize the embedding generator.
        
        Args:
            data_folder: Path to folder containing person images
            output_folder: Path to save generated embeddings
        """
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Initialize OpenCV face detector
        self.face_detector = OpenCVFaceDetector()
        
        # Create embeddings storage
        self.embeddings = {}
        self.face_locations = {}
        self.person_metadata = {}
        
    def load_and_process_image(self, image_path: Path) -> Tuple[np.ndarray, List, List]:
        """
        Load and process an image to extract face encodings.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (image_array, face_locations, face_encodings)
        """
        try:
            # Load image
            image = self.face_detector.load_image_file(str(image_path))
            
            # Find face locations
            face_locations = self.face_detector.face_locations(image)
            
            if not face_locations:
                logger.warning(f"No faces found in {image_path}")
                return None, [], []
            
            # Get face encodings
            face_encodings = self.face_detector.face_encodings(image, face_locations)
            
            return image, face_locations, face_encodings
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None, [], []
    
    def generate_embeddings(self):
        """
        Generate embeddings for all images in the data folder.
        """
        logger.info(f"Starting embedding generation from {self.data_folder}")
        
        # Get all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.data_folder.glob(f"*{ext}"))
            image_files.extend(self.data_folder.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} image files")
        
        processed_count = 0
        skipped_count = 0
        
        for image_path in image_files:
            logger.info(f"Processing {image_path.name}")
            
            # Load and process image
            image, face_locations, face_encodings = self.load_and_process_image(image_path)
            
            if image is None or not face_encodings:
                skipped_count += 1
                continue
            
            # Store embeddings for each face found
            person_name = image_path.stem  # Use filename without extension as person name
            
            if person_name not in self.embeddings:
                self.embeddings[person_name] = []
                self.face_locations[person_name] = []
                self.person_metadata[person_name] = {
                    'original_file': str(image_path),
                    'total_faces': len(face_encodings),
                    'image_shape': image.shape
                }
            
            # Add all face encodings for this person
            for i, encoding in enumerate(face_encodings):
                self.embeddings[person_name].append(encoding)
                self.face_locations[person_name].append(face_locations[i])
            
            processed_count += 1
            logger.info(f"Processed {person_name} with {len(face_encodings)} face(s)")
        
        logger.info(f"Embedding generation complete. Processed: {processed_count}, Skipped: {skipped_count}")
        
        # Save embeddings
        self.save_embeddings()
        
        return processed_count, skipped_count
    
    def save_embeddings(self):
        """
        Save generated embeddings to files.
        """
        # Save embeddings as pickle file
        embeddings_file = self.output_folder / "person_embeddings.pkl"
        with open(embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        # Save face locations
        locations_file = self.output_folder / "face_locations.pkl"
        with open(locations_file, 'wb') as f:
            pickle.dump(self.face_locations, f)
        
        # Save metadata as JSON
        metadata_file = self.output_folder / "person_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.person_metadata, f, indent=2)
        
        # Create a summary file
        summary = {
            'total_persons': len(self.embeddings),
            'total_embeddings': sum(len(encodings) for encodings in self.embeddings.values()),
            'persons': list(self.embeddings.keys())
        }
        
        summary_file = self.output_folder / "embedding_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Embeddings saved to {self.output_folder}")
        logger.info(f"Total persons: {summary['total_persons']}")
        logger.info(f"Total face embeddings: {summary['total_embeddings']}")
    
    def load_embeddings(self):
        """
        Load previously generated embeddings.
        """
        embeddings_file = self.output_folder / "person_embeddings.pkl"
        locations_file = self.output_folder / "face_locations.pkl"
        metadata_file = self.output_folder / "person_metadata.json"
        
        if not all([embeddings_file.exists(), locations_file.exists(), metadata_file.exists()]):
            raise FileNotFoundError("Embedding files not found. Run generate_embeddings() first.")
        
        with open(embeddings_file, 'rb') as f:
            self.embeddings = pickle.load(f)
        
        with open(locations_file, 'rb') as f:
            self.face_locations = pickle.load(f)
        
        with open(metadata_file, 'r') as f:
            self.person_metadata = json.load(f)
        
        logger.info(f"Loaded embeddings for {len(self.embeddings)} persons")

def main():
    """
    Main function to generate embeddings.
    """
    # Initialize generator
    generator = PersonEmbeddingGenerator()
    
    # Generate embeddings
    processed, skipped = generator.generate_embeddings()
    
    print(f"\nEmbedding generation complete!")
    print(f"Processed: {processed} images")
    print(f"Skipped: {skipped} images")
    print(f"Embeddings saved to: {generator.output_folder}")

if __name__ == "__main__":
    main()
