# Gym Equipment Detection FastAPI

This FastAPI backend runs our Fine-tuned YOLOv8 model to detect gym equipment
from camera images and returns the equipment name for use in the
Flutter mobile application.

## Requirements
- Python 3.9+
- Laptop and mobile phone on same Wi-Fi network

# **************************************************************************************************
## Setup Instructions

### 1. Create and activate virtual environment
```bash
python -m venv venv

venv\Scripts\activate
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
# **************************************************************************************************


## API Endpoints
### GET /
Health check

### GET /classes
Returns class ID to equipment name mapping

### POST /detect
Detects gym equipment from an image

### Request
- multipart/form-data
- key: file (image)

### Response
{
  "detected": true,
  "class_id": 2,
  "equipment": "dumbbell",
  "confidence": 0.98
}

## Flutter Integration Notes

Use base URL:
http://<laptop-ip>:8000

Laptop IP can be found using:
ipconfig

** Server must be running for detection to work **