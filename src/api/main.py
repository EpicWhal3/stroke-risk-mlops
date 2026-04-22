from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import mlflow
import pandas as pd

app = FastAPI(title="Stroke Risk API", version="1.0.0")


class Patient(BaseModel):
    age: float = Field(..., ge=0, le=100)
    avg_glucose_level: float = Field(..., ge=50, le=300)
    bmi: float = Field(..., ge=10, le=60)
    gender: str = Field(..., regex="^(Male|Female|Other)$")
    ever_married: str = Field(..., regex="^(Yes|No)$")
    work_type: str
    Residence_type: str
    smoking_status: str
    hypertension: int = Field(0, ge=0, le=1)
    heart_disease: int = Field(0, ge=0, le=1)


@app.post("/predict")
def predict(patient: Patient):
    model = mlflow.sklearn.load_model("models:/stroke-risk-prod/1")
    preprocessor = joblib.load("models/preprocessor.joblib")

    df = pd.DataFrame([patient.model_dump()])
    X = preprocessor.transform(df)
    prob = float(model.predict_proba(X)[0, 1])

    risk = "High" if prob > 0.5 else "Medium" if prob > 0.2 else "Low"
    return {"stroke_probability": round(prob, 4), "risk_level": risk}
