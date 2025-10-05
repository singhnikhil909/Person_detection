#!/bin/bash
echo "Person Detection System - Linux/Mac Installation Script"
echo "======================================================"

echo ""
echo "Installing requirements..."
echo "This system now uses OpenCV-based face detection (no cmake required!)"
echo ""

echo "Installing basic requirements..."
pip install numpy opencv-python Pillow Flask Werkzeug

echo ""
echo "Installation complete!"
echo ""
echo "The system is now ready to use with OpenCV-based face detection."
echo "No cmake or face-recognition installation required!"
