# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="AltScore AI", description="MSME Credit Scoring")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "credit_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded from root folder")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class PredictionRequest(BaseModel):
    revenue: float
    electricity: float
    upi_txn: float
    receipts: float
    gst: str
    age: float
    topups: float

def engineer_features(data: PredictionRequest):
    features = {
        'log_monthly_revenue': np.log1p(data.revenue),
        'electricity_bill_avg': data.electricity,
        'upi_transactions': data.upi_txn,
        'digital_receipts': data.receipts,
        'gst_filed': int(data.gst == 'yes'),
        'shop_age_months': data.age,
        'digital_adoption_score': (data.upi_txn + data.receipts) / 2,
        'payment_consistency': data.electricity / (data.revenue + 1),
        'mobile_engagement': data.topups / 30
    }
    return np.array([list(features.values())])

@app.post("/api/predict")
async def predict_risk(request: PredictionRequest):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        X = engineer_features(request)
        prob = model.predict_proba(X)[0][1]

        if prob > 0.6:
            risk = "High Risk"
        elif prob > 0.3:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        score = int(300 + 600 * (1 - prob))
        score = np.clip(score, 300, 900)

        # FIX: Convert NumPy types to Python native types
        return {
            "credit_score": int(score),                    # np.int64 → int
            "risk_level": risk,
            "default_probability": round(float(prob), 3)   # np.float64 → float
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "AltScore FastAPI (Flask-compatible) is running!"}