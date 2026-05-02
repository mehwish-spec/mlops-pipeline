# MLOps Pipeline with Model Monitoring & CI/CD

## Overview
End-to-end MLOps pipeline featuring automated model training, real-time data drift detection, experiment tracking with MLflow, and CI/CD automation with GitHub Actions.

## Features
- Automated model training with RandomForest + MLflow experiment tracking
- Statistical data drift detection across features with ALERT/OK status
- REST API with prediction, retraining and monitoring endpoints
- CI/CD pipeline with GitHub Actions — auto test, train and build on every push
- 3/3 pytest test coverage
- Docker containerization

## Tech Stack
Python, Scikit-learn, MLflow, FastAPI, Evidently AI, Docker, GitHub Actions, pytest

## Project Structure
mlops-pipeline/
├── app/
│   ├── main.py          # FastAPI server
│   └── train.py         # Model training + MLflow
├── monitoring/
│   └── drift_report.py  # Drift detection
├── tests/
│   └── test_api.py      # pytest tests
├── .github/
│   └── workflows/
│       └── ci.yml       # CI/CD pipeline
├── Dockerfile
└── requirements.txt

## API Endpoints
- POST /predict — Run model inference
- POST /retrain — Trigger model retraining
- GET /drift — Check data drift
- GET /metrics — View monitoring reports
- GET /health — Health check
- GET /docs — Swagger UI

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python app/train.py

# Run drift detection
python monitoring/drift_report.py

# Start API server
uvicorn app.main:app --port 8006

# Run tests
pytest tests/ -v
```

## Example
```bash
# Predict
curl -X POST http://localhost:8006/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 0.1, -0.3, 0.8, 0.2, 0.6, -0.1, 0.4, 0.7, -0.2]}'

# Response
{"prediction": 1, "probability": 0.51, "model_version": "v1.0"}

# Check drift
curl http://localhost:8006/drift

# Response
{"drift_detected": true, "drift_share": 0.2, "status": "ALERT"}
```

## MLflow Tracking
```bash
mlflow ui
# Open http://localhost:5000
```

## CI/CD Pipeline
Every push to main triggers:
1. Install dependencies
2. Train model
3. Run drift check
4. Run pytest tests
5. Build Docker image
