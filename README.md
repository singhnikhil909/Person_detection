# Person Detection System

A modular person detection and identification system that can detect and identify multiple persons in images using pre-generated face embeddings.

## Features

- **Modular Architecture**: Separate embedding generation and detection modules
- **Server Deployment**: Flask API for easy server deployment
- **Multiple Detection Methods**: File upload and base64 image support
- **Docker Support**: Containerized deployment with Docker
- **Unknown Person Handling**: Tags unknown persons as "Unknown"
- **Confidence Scoring**: Provides confidence scores for each detection

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Folder   │───▶│ Embedding        │───▶│ Detection       │
│   (Person       │    │ Generator        │    │ Module          │
│    Images)      │    │ (One-time)       │    │ (Server)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Install Dependencies

**For Windows (if you get dlib errors):**
```bash
# Run the installation script
install_requirements.bat

# Or use conda (recommended for Windows)
conda create -n person-detection python=3.9
conda activate person-detection
conda install -c conda-forge dlib
pip install face-recognition numpy opencv-python Pillow Flask Werkzeug
```

**For Linux/Mac:**
```bash
# Run the installation script
chmod +x install_requirements.sh
./install_requirements.sh

# Or install manually
pip install -r requirements.txt
pip install face-recognition
```

**If face-recognition installation fails, see [install_alternatives.md](install_alternatives.md) for more options.**

### 2. Generate Embeddings

Create embeddings from your person images:

```bash
# Generate embeddings from data folder
python generate_embeddings.py
```

This creates an `embeddings` folder with pre-computed face encodings.

### 3. Deploy Detection System

Start the detection server:

```bash
# Direct deployment
python deploy_detector.py

# Or with Docker
docker-compose up -d
```

The API will be available at `http://localhost:5000`

### 4. Test the System

```bash
python test_detection.py
```

## API Usage

### Upload Image for Detection

```bash
curl -X POST -F "image=@test_image.jpg" -F "tolerance=0.6" http://localhost:5000/detect
```

### Send Base64 Image

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"image_data": "base64_encoded_image", "tolerance": 0.6}' \
  http://localhost:5000/detect_base64
```

### Check System Status

```bash
curl http://localhost:5000/status
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API documentation |
| GET | `/status` | System status and statistics |
| POST | `/detect` | Upload image file for detection |
| POST | `/detect_base64` | Send base64 encoded image |

## Response Format

```json
{
  "success": true,
  "persons_found": 2,
  "detections": [
    {
      "person_name": "person_001",
      "confidence": 0.95,
      "face_location": {
        "top": 100,
        "right": 200,
        "bottom": 300,
        "left": 50
      },
      "face_index": 0
    },
    {
      "person_name": "Unknown",
      "confidence": 0.0,
      "face_location": {
        "top": 150,
        "right": 250,
        "bottom": 350,
        "left": 100
      },
      "face_index": 1
    }
  ],
  "message": "Found 2 person(s) in the image"
}
```

## File Structure

```
Person_detection/
├── data/                          # Person images (development only)
├── embeddings/                    # Generated embeddings (deployed)
│   ├── person_embeddings.pkl
│   ├── face_locations.pkl
│   ├── person_metadata.json
│   └── embedding_summary.json
├── embedding_generator.py         # Embedding generation module
├── person_detector.py            # Detection API module
├── deploy_detector.py            # Deployment script
├── generate_embeddings.py        # Embedding generation script
├── test_detection.py             # Testing script
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
├── docker-compose.yml           # Docker Compose configuration
├── deployment_guide.md          # Detailed deployment guide
└── README.md                     # This file
```

## Deployment Options

### 1. Direct Python Deployment

```bash
# Generate embeddings (one-time)
python generate_embeddings.py

# Deploy detection system
python deploy_detector.py
```

### 2. Docker Deployment

```bash
# Generate embeddings first
python generate_embeddings.py

# Deploy with Docker
docker-compose up -d
```

### 3. Production Deployment

See [deployment_guide.md](deployment_guide.md) for detailed production deployment instructions.

## Configuration

### Face Recognition Tolerance

- **0.4-0.5**: Strict matching (fewer false positives)
- **0.6**: Default (balanced)
- **0.7-0.8**: Lenient matching (may include false positives)

### Environment Variables

- `FLASK_ENV`: Set to `production` for production
- `FLASK_DEBUG`: Set to `False` for production

## Requirements

- Python 3.7+
- OpenCV
- face_recognition
- Flask
- NumPy
- Pillow

See `requirements.txt` for complete list.

## Troubleshooting

### Common Issues

1. **"Embedding files not found"**
   - Run `python generate_embeddings.py` first
   - Ensure embeddings folder exists with required files

2. **"No faces detected"**
   - Check image quality and lighting
   - Ensure faces are clearly visible
   - Try adjusting tolerance parameter

3. **"Detector not initialized"**
   - Check embeddings folder permissions
   - Verify all required files exist

### Performance Tips

1. **GPU Acceleration**: Use CUDA-enabled face_recognition
2. **Caching**: Implement Redis for frequently accessed embeddings
3. **Load Balancing**: Use multiple instances with load balancer

## Development

### Adding New Persons

1. Add new person images to `data` folder
2. Run `python generate_embeddings.py` to update embeddings
3. Restart detection system

### Customizing Detection

- Modify tolerance values for different accuracy requirements
- Adjust confidence thresholds
- Add custom face preprocessing

## Security Considerations

- Use HTTPS in production
- Implement API authentication if needed
- Add rate limiting for production use
- Validate uploaded images

## License

This project is open source. Please ensure you have proper rights to use the person images and comply with privacy regulations.

## Support

For issues and questions:
1. Check the logs for error messages
2. Verify all dependencies are installed
3. Ensure embeddings are properly generated
4. Test with the provided test script

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request
