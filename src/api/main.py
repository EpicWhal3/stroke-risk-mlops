import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Stroke Risk API", version="1.0.0")

model = None
preprocessor = None


def load_artifacts():
    global model, preprocessor
    if model is None:
        model_path = os.getenv("MODEL_PATH", "models/model.pkl")
        preprocessor_path = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.joblib")

        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found: {model_path}")
        if not os.path.exists(preprocessor_path):
            raise RuntimeError(f"Preprocessor file not found: {preprocessor_path}")

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)


class Patient(BaseModel):
    age: float = Field(..., ge=0, le=120)
    avg_glucose_level: float = Field(..., ge=0, le=500)
    bmi: float = Field(..., ge=0, le=100)
    gender: str
    ever_married: str
    work_type: str
    Residence_type: str
    smoking_status: str
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: Patient):
    try:
        load_artifacts()
        df = pd.DataFrame([patient.model_dump()])
        X = preprocessor.transform(df)
        prob = float(model.predict_proba(X)[0, 1])

        risk = "High" if prob >= 0.5 else "Medium" if prob >= 0.2 else "Low"

        return {
            "stroke_probability": round(prob, 4),
            "risk_level": risk,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
