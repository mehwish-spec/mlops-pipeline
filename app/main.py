from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import mlflow
import json
import os
from datetime import datetime
from monitoring.drift_report import check_drift
from app.train import generate_data, train_model

app = FastAPI(title="MLOps Pipeline API")

model = None

def load_model():
    global model
    if os.path.exists("models/model.pkl"):
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)
        print("Model loaded!")

@app.on_event("startup")
async def startup():
    load_model()

class PredictRequest(BaseModel):
    features: list

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if model is None:
        load_model()
    X = pd.DataFrame([req.features], columns=[f"feature_{i}" for i in range(len(req.features))])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()
    return PredictResponse(
        prediction=int(pred),
        probability=round(float(prob), 4),
        model_version="v1.0"
    )

@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    def retrain_task():
        print("Retraining model...")
        df = generate_data(n_samples=1000)
        train_model(df)
        load_model()
        print("Retraining complete!")
    background_tasks.add_task(retrain_task)
    return {"message": "Retraining started in background"}

@app.get("/drift")
async def drift_check():
    if not os.path.exists("data/reference_data.csv"):
        return {"error": "No reference data found. Train model first."}
    result = check_drift()
    return result

@app.get("/metrics")
async def get_metrics():
    reports = []
    if os.path.exists("monitoring/reports"):
        for f in sorted(os.listdir("monitoring/reports")):
            if f.startswith("summary_") and f.endswith(".json"):
                with open("monitoring/reports/" + f) as rf:
                    reports.append(json.load(rf))
    return {"reports": reports[-5:]}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }
