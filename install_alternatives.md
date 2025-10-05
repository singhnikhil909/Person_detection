# Alternative Installation Methods for face-recognition

The `face-recognition` library depends on `dlib`, which can be difficult to install on Windows. Here are several alternative approaches:

## Method 1: Use Pre-compiled Wheels (Recommended)

### Windows
```bash
# Try installing from a different source
pip install --find-links https://github.com/ageitgey/face_recognition_models/releases/download/v0.3.0/ face_recognition

# Or try with specific Python version
pip install --only-binary=all face-recognition
```

### Linux/Mac
```bash
# Usually works on Linux/Mac
pip install face-recognition
```

## Method 2: Use Conda (Recommended for Windows)

If you have Anaconda or Miniconda installed:

```bash
# Create a new environment
conda create -n person-detection python=3.9

# Activate environment
conda activate person-detection

# Install dlib from conda-forge
conda install -c conda-forge dlib

# Install face-recognition
pip install face-recognition

# Install other requirements
pip install numpy opencv-python Pillow Flask Werkzeug
```

## Method 3: Use Docker (Easiest)

If you're having trouble with local installation, use Docker:

```bash
# Build and run with Docker
docker-compose up -d
```

## Method 4: Manual dlib Installation (Windows)

### Prerequisites:
1. Install Visual Studio Build Tools
2. Install CMake
3. Install Git

### Steps:
```bash
# Install CMake
pip install cmake

# Install dlib from source
pip install dlib

# Install face-recognition
pip install face-recognition
```

## Method 5: Use Alternative Face Recognition Library

If `face-recognition` continues to fail, you can modify the code to use OpenCV's built-in face detection:

### Install OpenCV with face detection:
```bash
pip install opencv-contrib-python
```

### Modified requirements.txt:
```txt
numpy>=1.21.0
opencv-contrib-python>=4.5.0
Pillow>=8.3.0
Flask>=2.0.0
Werkzeug>=2.0.0
```

## Method 6: Use Google Colab

For testing and development, you can use Google Colab which has `face-recognition` pre-installed:

1. Upload your code to Google Colab
2. Run the embedding generation
3. Download the embeddings folder
4. Deploy locally with the embeddings

## Method 7: Use Cloud Services

Deploy directly to cloud platforms that handle dependencies:

### Heroku:
```bash
# Add to requirements.txt
face-recognition>=1.3.0

# Deploy
git push heroku main
```

### AWS/GCP/Azure:
Use containerized deployment with Docker.

## Troubleshooting

### Common Issues:

1. **"Microsoft Visual C++ 14.0 is required"**
   - Install Visual Studio Build Tools
   - Or use conda instead

2. **"CMake not found"**
   - Install CMake: `pip install cmake`
   - Or download from cmake.org

3. **"Failed building wheel for dlib"**
   - Try conda: `conda install -c conda-forge dlib`
   - Or use Docker

4. **"No module named 'dlib'"**
   - Ensure dlib installed correctly
   - Check Python environment

### Verification:

Test if face-recognition is working:
```python
import face_recognition
print("face-recognition installed successfully!")
```

## Recommended Approach

For **Windows users**, I recommend:
1. **First try**: Use conda (Method 2)
2. **If that fails**: Use Docker (Method 3)
3. **For production**: Use cloud deployment

For **Linux/Mac users**:
1. **First try**: Direct pip install
2. **If that fails**: Use conda
3. **For production**: Use Docker

## Quick Start with Conda (Windows)

```bash
# Install Miniconda from https://docs.conda.io/en/latest/miniconda.html

# Create environment
conda create -n person-detection python=3.9
conda activate person-detection

# Install dlib
conda install -c conda-forge dlib

# Install other packages
pip install face-recognition numpy opencv-python Pillow Flask Werkzeug

# Run the system
python generate_embeddings.py
python deploy_detector.py
```
