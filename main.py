# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="AltScore AI", description="MSME Credit Scoring")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === LOAD YOUR MODEL ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "credit_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# === INPUT SCHEMA ===
class PredictionRequest(BaseModel):
    monthly_revenue: float
    electricity_bill: float
    upi_volume: float
    digital_rec: float
    gst_registered: int  # 1 = Yes, 0 = No
    business_age: float
    outstanding_loans: float

# === PREDICTION LOGIC ===
def get_features(data: PredictionRequest):
    return np.array([[
        data.monthly_revenue,
        data.electricity_bill,
        data.upi_volume,
        data.digital_rec,
        data.gst_registered,
        data.business_age,
        data.outstanding_loans
    ]])

@app.post("/api/predict")
async def predict_risk(request: PredictionRequest):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        features = get_features(request)
        
        # If your model outputs probability of default
        default_prob = model.predict_proba(features)[0][1]  # class 1 = default
        
        # Convert to credit score (300–900)
        score = int(300 + 600 * (1 - default_prob))
        score = np.clip(score, 300, 900)

        # Risk level
        if score >= 750:
            risk = "Low"
        elif score >= 600:
            risk = "Medium"
        else:
            risk = "High"

        return {
            "credit_score": score,
            "risk_level": risk,
            "default_probability": round(default_prob, 3)
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "AltScore FastAPI + Your Model = Running!"}