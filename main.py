# main.py
from fastapi import FastAPI, HTTPException
import requests as request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import re
app = FastAPI(title="AltScore AI", description="MSME Credit Scoring")

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

with open("context.txt", "r", encoding="utf-8") as f:
    CONTEXT_PROMPT = f.read()


for m in genai.list_models():
    print(m.name)

# CORS
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:5173"],
    allow_origins=["*"],
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
    
    
class ChatRequest(BaseModel):
    message: str
    
    


# @app.post("/chat")
# async def chat(req: ChatRequest):
#     try:
#         print("🔹 Request received:", req.message)

#         url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}"
#         payload = {
#             "contents": [
#                 {"parts": [{"text": CONTEXT_PROMPT}]},  # Context from file
#                 {"parts": [{"text": req.message}]}       # User input
#             ]
#         }

#         response = request.post(url, json=payload)
#         data = response.json()
#         reply = data["candidates"][0]["content"]["parts"][0]["text"]

#         print("🔹 Gemini response received.")
#         return {"reply": reply}

#     except Exception as e:
#         print("Error:", e)
#         return {"error": str(e)}

    
# GENERATION_CONFIG = genai.GenerativeModel(
#     # model_name="gemini-2.5-flash",
#     generation_config={
#         "max_output_tokens": 150, 
#         "temperature": 0.4,        
#         "top_p": 0.8,              
#         "top_k": 40,               
#     }
# )

def clean_markdown(text: str) -> str:
    """Remove markdown symbols like **, *, _, and #."""
    return re.sub(r"[*_#`]+", "", text)

GENERATION_CONFIG = {
    "maxOutputTokens": 150,
    "temperature": 0.4,
    "topP": 0.8,
    "topK": 40
}

MODEL_NAME = "gemini-2.5-flash-lite"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
    f"?key={GEMINI_API_KEY}"
)


@app.post("/chat")
async def chat(req: ChatRequest):
    """Send user message + MSME context to Gemini model."""
    print(f"🔹 Request received: {req.message}")

    try:
        payload = {
            "contents": [
                {"parts": [{"text": CONTEXT_PROMPT}]},
                {"parts": [{"text": req.message}]}
            ],
            "generationConfig": GENERATION_CONFIG
        }

        response = request.post(GEMINI_URL, json=payload)
        response.raise_for_status()

        data = response.json()
        reply = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )

        if not reply:
            raise ValueError("Empty response from Gemini API")

        cleaned_reply = clean_markdown(reply.strip())

        print("✅ Gemini response received successfully.")
        return {"reply": cleaned_reply}

    except request.exceptions.RequestException as e:
        print(f"Network or API error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini API request failed: {str(e)}")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.get("/")
async def root():
    return {"message": "AltScore FastAPI + Your Model = Running!"}