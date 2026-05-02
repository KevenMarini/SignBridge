# SignBridge

An AI-based Sign Language Detection and Communication System designed to bridge the gap between sign language users and others.

## Features
- Real-time gesture recognition using Mediapipe.
- Scalable backend for processing and data management.
- User-friendly frontend interface.

## Project Structure
- `backend/`: Flask/FastAPI backend logic.
- `frontend/`: Web-based interface.
- `realtime_detection.py`: Core script for camera-based detection.
- `train_model.py`: Script for training the gesture recognition model.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the detection: `python realtime_detection.py`
