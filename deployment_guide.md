# Person Detection System - Deployment Guide

This guide explains how to deploy the person detection system on a server.

## System Architecture

The system consists of two main components:

1. **Embedding Generator** (`embedding_generator.py`) - Creates embeddings from person images
2. **Person Detector** (`person_detector.py`) - Detects and identifies persons using pre-generated embeddings

## Prerequisites

- Python 3.7+
- Required Python packages (see `requirements.txt`)
- Images of persons in the `data` folder

## Step 1: Generate Embeddings

Before deploying, you need to generate embeddings from your person images:

```bash
# Install dependencies
pip install -r requirements.txt

# Generate embeddings from data folder
python generate_embeddings.py
```

This will create an `embeddings` folder with:
- `person_embeddings.pkl` - Face encodings for each person
- `face_locations.pkl` - Face location data
- `person_metadata.json` - Metadata about each person
- `embedding_summary.json` - Summary of generated embeddings

## Step 2: Deploy Detection System

### Option A: Direct Python Deployment

```bash
# Start the detection server
python deploy_detector.py
```

The server will be available at `http://localhost:5000`

### Option B: Docker Deployment

1. **Build Docker image:**
```bash
docker build -t person-detector .
```

2. **Run with Docker Compose:**
```bash
docker-compose up -d
```

3. **Or run directly:**
```bash
docker run -p 5000:5000 -v ./embeddings:/app/embeddings:ro person-detector
```

## API Endpoints

### GET `/`
- Returns API documentation page

### GET `/status`
- Check system status and embedding information
- Response includes total persons and face encodings loaded

### POST `/detect`
- Upload image file for person detection
- Parameters:
  - `image`: Image file (multipart/form-data)
  - `tolerance`: Face recognition tolerance (optional, default: 0.6)

### POST `/detect_base64`
- Send base64 encoded image for detection
- Parameters:
  - `image_data`: Base64 encoded image string
  - `tolerance`: Face recognition tolerance (optional, default: 0.6)

## Testing the System

Run the test script to verify the API:

```bash
python test_detection.py
```

## Production Deployment

### Using Docker (Recommended)

1. **Generate embeddings on your development machine:**
```bash
python generate_embeddings.py
```

2. **Copy embeddings to server:**
```bash
scp -r embeddings/ user@server:/path/to/deployment/
```

3. **Deploy on server:**
```bash
# On server
docker-compose up -d
```

### Using Systemd (Linux)

1. **Create service file** `/etc/systemd/system/person-detector.service`:
```ini
[Unit]
Description=Person Detection API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/deployment
ExecStart=/usr/bin/python3 deploy_detector.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

2. **Enable and start service:**
```bash
sudo systemctl enable person-detector
sudo systemctl start person-detector
```

## Configuration

### Environment Variables

- `FLASK_ENV`: Set to `production` for production deployment
- `FLASK_DEBUG`: Set to `False` for production

### Face Recognition Tolerance

- Lower values (0.4-0.5): More strict matching, fewer false positives
- Higher values (0.6-0.8): More lenient matching, may include false positives
- Default: 0.6

## Monitoring

### Health Checks

The system includes health check endpoints:
- `GET /status` - Returns system status and statistics

### Logs

Monitor logs for errors and performance:
```bash
# Docker logs
docker-compose logs -f

# Systemd logs
journalctl -u person-detector -f
```

## Security Considerations

1. **Network Security**: Use reverse proxy (nginx) with SSL
2. **Authentication**: Add API authentication if needed
3. **Rate Limiting**: Implement rate limiting for production use
4. **Input Validation**: Validate uploaded images

## Troubleshooting

### Common Issues

1. **"Embedding files not found"**
   - Ensure embeddings folder exists with required files
   - Run `python generate_embeddings.py` first

2. **"No faces detected"**
   - Check image quality and lighting
   - Ensure faces are clearly visible
   - Try adjusting tolerance parameter

3. **"Detector not initialized"**
   - Check if embeddings folder is accessible
   - Verify all required files exist
   - Check file permissions

### Performance Optimization

1. **GPU Acceleration**: Use CUDA-enabled face_recognition for better performance
2. **Caching**: Implement Redis caching for frequently accessed embeddings
3. **Load Balancing**: Use multiple instances with load balancer

## File Structure

```
Person_detection/
├── data/                          # Person images (not deployed)
├── embeddings/                    # Generated embeddings (deployed)
│   ├── person_embeddings.pkl
│   ├── face_locations.pkl
│   ├── person_metadata.json
│   └── embedding_summary.json
├── embedding_generator.py         # Embedding generation (development)
├── person_detector.py            # Detection API (deployed)
├── deploy_detector.py            # Deployment script
├── generate_embeddings.py        # Embedding generation script
├── test_detection.py             # Testing script
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
├── docker-compose.yml           # Docker Compose configuration
└── deployment_guide.md          # This guide
```

## Support

For issues and questions:
1. Check the logs for error messages
2. Verify all dependencies are installed
3. Ensure embeddings are properly generated
4. Test with the provided test script
