"""
OpenCV-based Face Detection and Recognition Module
Alternative to face_recognition that doesn't require cmake
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class OpenCVFaceDetector:
    """
    OpenCV-based face detection and recognition system.
    Uses OpenCV DNN module with pre-trained models for face detection and recognition.
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the OpenCV face detector.
        
        Args:
            confidence_threshold: Minimum confidence for face detection
        """
        self.confidence_threshold = confidence_threshold
        self.face_net = None
        self.face_recognizer = None
        self.known_faces = {}
        self.face_labels = []
        
        # Initialize face detection model
        self._initialize_face_detection()
        
        # Initialize face recognition
        self._initialize_face_recognition()
    
    def _initialize_face_detection(self):
        """Initialize OpenCV DNN face detection model."""
        try:
            # Use OpenCV's built-in DNN face detection
            # This uses a pre-trained model that comes with OpenCV
            model_path = self._get_face_detection_model()
            if model_path and os.path.exists(model_path):
                self.face_net = cv2.dnn.readNetFromTensorflow(model_path)
                logger.info("Face detection model loaded successfully")
            else:
                # Fallback to Haar Cascade
                self.face_net = None
                logger.info("Using Haar Cascade for face detection")
        except Exception as e:
            logger.warning(f"Could not load DNN model, using Haar Cascade: {e}")
            self.face_net = None
    
    def _get_face_detection_model(self) -> Optional[str]:
        """Get path to face detection model."""
        # Try to find OpenCV's built-in face detection model
        opencv_data = cv2.data.haarcascades
        model_path = os.path.join(opencv_data, "..", "dnn", "opencv_face_detector_uint8.pb")
        if os.path.exists(model_path):
            return model_path
        
        # Alternative model path
        model_path = os.path.join(opencv_data, "..", "dnn", "opencv_face_detector.pb")
        if os.path.exists(model_path):
            return model_path
            
        return None
    
    def _initialize_face_recognition(self):
        """Initialize face recognition using histogram-based features."""
        try:
            # Try to use LBPH face recognizer if opencv-contrib-python is available
            if hasattr(cv2, 'face'):
                self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
                logger.info("Face recognition using LBPH (opencv-contrib-python available)")
            else:
                # Fallback to histogram-based features
                self.face_recognizer = None
                logger.info("Face recognition using histogram-based features")
        except Exception as e:
            logger.warning(f"Face recognition initialization: {e}")
            self.face_recognizer = None
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of face bounding boxes (x, y, width, height)
        """
        if self.face_net is not None:
            return self._detect_faces_dnn(image)
        else:
            return self._detect_faces_haar(image)
    
    def _detect_faces_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using DNN model."""
        try:
            h, w = image.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > self.confidence_threshold:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    # Convert to (x, y, width, height) format
                    faces.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            
            return faces
        except Exception as e:
            logger.error(f"DNN face detection failed: {e}")
            return self._detect_faces_haar(image)
    
    def _detect_faces_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar Cascade."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Load Haar Cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Ensure faces is a list of tuples
            if len(faces) == 0:
                return []
            
            # Convert to list of tuples
            face_list = []
            for (x, y, w, h) in faces:
                face_list.append((int(x), int(y), int(w), int(h)))
            
            return face_list
        except Exception as e:
            logger.error(f"Haar face detection failed: {e}")
            return []
    
    def extract_face_features(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract features from a face region.
        
        Args:
            image: Input image
            face_box: Face bounding box (x, y, width, height)
            
        Returns:
            Face features as numpy array
        """
        try:
            x, y, w, h = face_box
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, image.shape[1]))
            y = max(0, min(y, image.shape[0]))
            w = max(1, min(w, image.shape[1] - x))
            h = max(1, min(h, image.shape[0] - y))
            
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            
            if face_region.size == 0:
                logger.warning("Empty face region, returning zero features")
                return np.array([0.0] * 262)
            
            # Resize to standard size
            face_region = cv2.resize(face_region, (100, 100))
            
            # Convert to grayscale if needed
            if len(face_region.shape) == 3:
                face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Extract features using simple histogram
            features = self._extract_histogram_features(face_region)
            
            # Ensure features is a numpy array
            if not isinstance(features, np.ndarray):
                logger.warning("Features not numpy array, converting")
                features = np.array(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting face features: {e}")
            # Return a zero feature vector as fallback
            return np.array([0.0] * 262)
    
    def _extract_histogram_features(self, face_image: np.ndarray) -> np.ndarray:
        """Extract histogram-based features from face image."""
        try:
            # Calculate histogram
            hist = cv2.calcHist([face_image], [0], None, [256], [0, 256])
            
            # Normalize histogram
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)
            
            # Add statistical features - ensure they are scalars
            mean_val = float(np.mean(face_image))
            std_val = float(np.std(face_image))
            min_val = float(np.min(face_image))
            max_val = float(np.max(face_image))
            
            # Add texture features using Local Binary Pattern-like approach
            # Calculate gradients
            grad_x = cv2.Sobel(face_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Statistical features of gradients - ensure they are scalars
            grad_mean = float(np.mean(gradient_magnitude))
            grad_std = float(np.std(gradient_magnitude))
            
            # Combine all features - ensure all are numpy arrays
            features = np.concatenate([
                hist, 
                np.array([mean_val, std_val, min_val, max_val]),
                np.array([grad_mean, grad_std])
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting histogram features: {e}")
            # Return a simple feature vector as fallback
            return np.array([0.0] * 262)  # 256 (hist) + 4 (stats) + 2 (grad) = 262
    
    def _compare_two_faces(self, face_features1: np.ndarray, face_features2: np.ndarray) -> float:
        """
        Compare two face feature vectors.
        
        Args:
            face_features1: First face features
            face_features2: Second face features
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        try:
            logger.info(f"Comparing faces: type1={type(face_features1)}, type2={type(face_features2)}")
            logger.info(f"Shapes: {face_features1.shape if hasattr(face_features1, 'shape') else 'no shape'}, {face_features2.shape if hasattr(face_features2, 'shape') else 'no shape'}")
            
            if len(face_features1) == 0 or len(face_features2) == 0:
                logger.warning("Empty face features")
                return 0.0
            
            # Ensure same length
            min_len = min(len(face_features1), len(face_features2))
            features1 = face_features1[:min_len]
            features2 = face_features2[:min_len]
            
            # Calculate cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                logger.warning("Zero norm in face comparison")
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Convert to 0-1 range
            similarity = max(0, min(1, (similarity + 1) / 2))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error in _compare_two_faces: {e}")
            return 0.0
    
    def load_image_file(self, image_path: str) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image
    
    def face_locations(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Find face locations in image.
        
        Args:
            image: Input image
            
        Returns:
            List of face locations (top, right, bottom, left)
        """
        faces = self.detect_faces(image)
        
        # Ensure faces is a list
        if not isinstance(faces, list):
            logger.warning(f"detect_faces returned {type(faces)}, expected list")
            return []
        
        # Convert from (x, y, w, h) to (top, right, bottom, left)
        face_locations = []
        for face in faces:
            if len(face) == 4:
                x, y, w, h = face
                top = int(y)
                right = int(x + w)
                bottom = int(y + h)
                left = int(x)
                face_locations.append((top, right, bottom, left))
            else:
                logger.warning(f"Invalid face format: {face}")
        
        return face_locations
    
    def face_encodings(self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """
        Get face encodings for detected faces.
        
        Args:
            image: Input image
            face_locations: List of face locations
            
        Returns:
            List of face encodings
        """
        encodings = []
        
        try:
            for i, face_location in enumerate(face_locations):
                try:
                    # Ensure face_location is a tuple with 4 elements
                    if len(face_location) != 4:
                        logger.warning(f"Invalid face location format: {face_location}")
                        continue
                    
                    top, right, bottom, left = face_location
                    
                    # Convert back to (x, y, w, h) format
                    x, y, w, h = left, top, right - left, bottom - top
                    
                    # Extract features
                    features = self.extract_face_features(image, (x, y, w, h))
                    
                    # Ensure features is a numpy array
                    if isinstance(features, np.ndarray):
                        encodings.append(features)
                    else:
                        logger.warning(f"Invalid features type: {type(features)}")
                        encodings.append(np.array([0.0] * 262))
                        
                except Exception as e:
                    logger.error(f"Error processing face {i}: {e}")
                    # Add a zero feature vector as fallback
                    encodings.append(np.array([0.0] * 262))
            
            return encodings
            
        except Exception as e:
            logger.error(f"Error in face_encodings: {e}")
            return []
    
    def compare_faces(self, known_encodings: List[np.ndarray], face_encoding: np.ndarray, tolerance: float = 0.6) -> List[bool]:
        """
        Compare a face encoding with known encodings.
        
        Args:
            known_encodings: List of known face encodings
            face_encoding: Face encoding to compare
            tolerance: Similarity threshold
            
        Returns:
            List of boolean matches
        """
        matches = []
        
        logger.info(f"Comparing face encoding (type: {type(face_encoding)}) with {len(known_encodings)} known encodings")
        
        for i, known_encoding in enumerate(known_encodings):
            try:
                logger.info(f"Comparing with known encoding {i}, type: {type(known_encoding)}")
                similarity = self._compare_two_faces(known_encoding, face_encoding)
                matches.append(similarity >= tolerance)
            except Exception as e:
                logger.error(f"Error comparing with known encoding {i}: {e}")
                matches.append(False)
        
        return matches
    
    def face_distance(self, face_encodings: List[np.ndarray], face_to_check: np.ndarray) -> List[float]:
        """
        Calculate distances between face encodings.
        
        Args:
            face_encodings: List of known face encodings
            face_to_check: Face encoding to check
            
        Returns:
            List of distances (lower is more similar)
        """
        distances = []
        
        for encoding in face_encodings:
            similarity = self._compare_two_faces(encoding, face_to_check)
            # Convert similarity to distance (1 - similarity)
            distance = 1 - similarity
            distances.append(distance)
        
        return distances
