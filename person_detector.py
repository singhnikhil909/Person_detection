"""
Person Detection System - Detection Module
This module detects and identifies persons in input images using pre-generated embeddings.
This is the module that will be deployed on the server.
"""

import os
import pickle
import numpy as np
import cv2
from PIL import Image
from opencv_face_detector import OpenCVFaceDetector
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging
from flask import Flask, request, jsonify, render_template_string, render_template, send_file
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonDetector:
    def __init__(self, embeddings_folder: str = "embeddings"):
        """
        Initialize the person detector with pre-generated embeddings.
        
        Args:
            embeddings_folder: Path to folder containing pre-generated embeddings
        """
        self.embeddings_folder = Path(embeddings_folder)
        self.embeddings = {}
        self.face_locations = {}
        self.person_metadata = {}
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Initialize OpenCV face detector
        self.face_detector = OpenCVFaceDetector()
        
        # Load embeddings
        self.load_embeddings()
        
    def load_embeddings(self):
        """
        Load pre-generated embeddings from files.
        """
        embeddings_file = self.embeddings_folder / "person_embeddings.pkl"
        locations_file = self.embeddings_folder / "face_locations.pkl"
        metadata_file = self.embeddings_folder / "person_metadata.json"
        
        if not all([embeddings_file.exists(), locations_file.exists(), metadata_file.exists()]):
            raise FileNotFoundError(f"Embedding files not found in {self.embeddings_folder}. Run embedding_generator.py first.")
        
        with open(embeddings_file, 'rb') as f:
            self.embeddings = pickle.load(f)
        
        with open(locations_file, 'rb') as f:
            self.face_locations = pickle.load(f)
        
        with open(metadata_file, 'r') as f:
            self.person_metadata = json.load(f)
        
        # Create flat lists for face_recognition comparison
        for person_name, encodings in self.embeddings.items():
            for encoding in encodings:
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(person_name)
        
        logger.info(f"Loaded embeddings for {len(self.embeddings)} persons")
        logger.info(f"Total face encodings: {len(self.known_face_encodings)}")
    
    def detect_persons_in_image(self, image_path: str, tolerance: float = 0.6) -> Dict:
        """
        Detect and identify persons in an image.
        
        Args:
            image_path: Path to the input image
            tolerance: Face recognition tolerance (lower = more strict)
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Load image
            image = self.face_detector.load_image_file(image_path)
            
            # Find face locations
            face_locations = self.face_detector.face_locations(image)
            
            # Ensure face_locations is a list
            if not isinstance(face_locations, list):
                logger.error(f"face_locations returned {type(face_locations)}, expected list")
                return {
                    'success': False,
                    'persons_found': 0,
                    'detections': [],
                    'message': 'Error in face detection'
                }
            
            if not face_locations:
                return {
                    'success': True,
                    'persons_found': 0,
                    'detections': [],
                    'message': 'No faces detected in the image'
                }
            
            # Get face encodings
            face_encodings = self.face_detector.face_encodings(image, face_locations)
            logger.info(f"Got {len(face_encodings)} face encodings")
            
            detections = []
            
            for i, face_encoding in enumerate(face_encodings):
                logger.info(f"Processing face {i}, encoding type: {type(face_encoding)}")
                # Compare with known faces
                matches = self.face_detector.compare_faces(
                    self.known_face_encodings, 
                    face_encoding, 
                    tolerance=tolerance
                )
                
                face_distances = self.face_detector.face_distance(
                    self.known_face_encodings, 
                    face_encoding
                )
                
                # Find best match
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    person_name = self.known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                else:
                    person_name = "Unknown"
                    confidence = 0.0
                
                # Get face location coordinates
                top, right, bottom, left = face_locations[i]
                
                detection = {
                    'person_name': person_name,
                    'confidence': float(confidence),
                    'face_location': {
                        'top': int(top),
                        'right': int(right),
                        'bottom': int(bottom),
                        'left': int(left)
                    },
                    'face_index': i
                }
                
                detections.append(detection)
            
            return {
                'success': True,
                'persons_found': len(detections),
                'detections': detections,
                'message': f'Found {len(detections)} person(s) in the image'
            }
            
        except Exception as e:
            logger.error(f"Error detecting persons in {image_path}: {str(e)}")
            return {
                'success': False,
                'persons_found': 0,
                'detections': [],
                'message': f'Error processing image: {str(e)}'
            }
    
    def detect_persons_from_array(self, image_array: np.ndarray, tolerance: float = 0.6) -> Dict:
        """
        Detect and identify persons from image array.
        
        Args:
            image_array: Image as numpy array
            tolerance: Face recognition tolerance (lower = more strict)
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Find face locations
            face_locations = self.face_detector.face_locations(image_array)
            
            # Ensure face_locations is a list
            if not isinstance(face_locations, list):
                logger.error(f"face_locations returned {type(face_locations)}, expected list")
                return {
                    'success': False,
                    'persons_found': 0,
                    'detections': [],
                    'message': 'Error in face detection'
                }
            
            if not face_locations:
                return {
                    'success': True,
                    'persons_found': 0,
                    'detections': [],
                    'message': 'No faces detected in the image'
                }
            
            # Get face encodings
            face_encodings = self.face_detector.face_encodings(image_array, face_locations)
            
            detections = []
            
            for i, face_encoding in enumerate(face_encodings):
                # Compare with known faces
                matches = self.face_detector.compare_faces(
                    self.known_face_encodings, 
                    face_encoding, 
                    tolerance=tolerance
                )
                
                face_distances = self.face_detector.face_distance(
                    self.known_face_encodings, 
                    face_encoding
                )
                
                # Find best match
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    person_name = self.known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                else:
                    person_name = "Unknown"
                    confidence = 0.0
                
                # Get face location coordinates
                top, right, bottom, left = face_locations[i]
                
                detection = {
                    'person_name': person_name,
                    'confidence': float(confidence),
                    'face_location': {
                        'top': int(top),
                        'right': int(right),
                        'bottom': int(bottom),
                        'left': int(left)
                    },
                    'face_index': i
                }
                
                detections.append(detection)
            
            return {
                'success': True,
                'persons_found': len(detections),
                'detections': detections,
                'message': f'Found {len(detections)} person(s) in the image'
            }
            
        except Exception as e:
            logger.error(f"Error detecting persons: {str(e)}")
            return {
                'success': False,
                'persons_found': 0,
                'detections': [],
                'message': f'Error processing image: {str(e)}'
            }
    
    def draw_detections(self, image_path: str, output_path: str, tolerance: float = 0.6) -> bool:
        """
        Draw bounding boxes around detected faces and save the result.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image
            tolerance: Face recognition tolerance
            
        Returns:
            Boolean indicating success
        """
        try:
            # Detect persons
            result = self.detect_persons_in_image(image_path, tolerance)
            
            if not result['success']:
                return False
            
            # Load image for drawing
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Draw bounding boxes
            for detection in result['detections']:
                top, right, bottom, left = detection['face_location'].values()
                person_name = detection['person_name']
                confidence = detection['confidence']
                
                # Choose color based on whether person is known
                color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
                
                # Draw rectangle
                cv2.rectangle(image, (left, top), (right, bottom), color, 2)
                
                # Draw label
                label = f"{person_name} ({confidence:.2f})"
                cv2.putText(image, label, (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save result
            cv2.imwrite(output_path, image)
            return True
            
        except Exception as e:
            logger.error(f"Error drawing detections: {str(e)}")
            return False

# Flask API for server deployment
app = Flask(__name__)

# Initialize detector
detector = None

def initialize_detector():
    """Initialize the person detector."""
    global detector
    try:
        detector = PersonDetector()
        logger.info("Person detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {str(e)}")
        detector = None

@app.route('/')
def index():
    """Home page with UI for person detection."""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_persons():
    """Detect persons in uploaded image."""
    if detector is None:
        return jsonify({'error': 'Detector not initialized'}), 500
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get tolerance parameter
        tolerance = float(request.form.get('tolerance', 0.6))
        
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        file.save(temp_path)
        
        try:
            # Detect persons
            result = detector.detect_persons_in_image(temp_path, tolerance)
            return jsonify(result)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error in detect_persons: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect_base64', methods=['POST'])
def detect_persons_base64():
    """Detect persons from base64 encoded image."""
    if detector is None:
        return jsonify({'error': 'Detector not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get tolerance parameter
        tolerance = float(data.get('tolerance', 0.6))
        
        # Decode base64 image
        image_data = data['image_data']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Convert to RGB for face_recognition
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect persons
        result = detector.detect_persons_from_array(image_rgb, tolerance)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in detect_persons_base64: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get API status and embedding information."""
    if detector is None:
        return jsonify({
            'status': 'error',
            'message': 'Detector not initialized',
            'embeddings_loaded': False
        }), 500
    
    try:
        return jsonify({
            'status': 'ready',
            'message': 'Detector initialized successfully',
            'embeddings_loaded': True,
            'total_persons': len(detector.embeddings),
            'total_face_encodings': len(detector.known_face_encodings),
            'persons': list(detector.embeddings.keys())
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'embeddings_loaded': False
        }), 500

@app.route('/detect_with_image', methods=['POST'])
def detect_persons_with_image():
    """Detect persons in uploaded image and return processed image with bounding boxes."""
    if detector is None:
        return jsonify({'error': 'Detector not initialized'}), 500
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get tolerance parameter
        tolerance = float(request.form.get('tolerance', 0.6))
        
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        file.save(temp_path)
        
        try:
            # Detect persons
            result = detector.detect_persons_in_image(temp_path, tolerance)
            
            if not result['success']:
                return jsonify(result), 400
            
            # Create image with bounding boxes
            image = cv2.imread(temp_path)
            if image is None:
                return jsonify({'error': 'Could not load image'}), 400
            
            # Draw bounding boxes on the image
            for detection in result['detections']:
                top, right, bottom, left = detection['face_location'].values()
                person_name = detection['person_name']
                confidence = detection['confidence']
                
                # Choose color based on whether person is known
                color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
                
                # Draw rectangle
                cv2.rectangle(image, (left, top), (right, bottom), color, 3)
                
                # Draw label
                label = f"{person_name} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Draw label background
                cv2.rectangle(image, (left, top - label_size[1] - 10), 
                             (left + label_size[0] + 10, top), color, -1)
                
                # Draw label text
                cv2.putText(image, label, (left + 5, top - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Encode image as base64
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Return result with image
            result['processed_image'] = f"data:image/jpeg;base64,{image_base64}"
            
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error in detect_persons_with_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize detector
    initialize_detector()
    
    if detector is not None:
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Failed to initialize detector. Exiting.")
