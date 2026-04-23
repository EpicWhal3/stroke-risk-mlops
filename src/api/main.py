import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Stroke Risk API", version="1.0.0")

model = joblib.load("models/model.pkl")
preprocessor = joblib.load("models/preprocessor.joblib")


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
    df = pd.DataFrame([patient.model_dump()])
    X = preprocessor.transform(df)
    prob = float(model.predict_proba(X)[0, 1])

    if prob >= 0.5:
        risk = "High"
    elif prob >= 0.2:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "stroke_probability": round(prob, 4),
        "risk_level": risk,
    }
