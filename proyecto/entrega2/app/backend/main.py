import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Columnas utilizadas durante el entrenamiento
SAFE_FEATURES = [
    "region_id",
    "customer_type",
    "Y",
    "X",
    "size",
    "brand",
    "category",
    "sub_category",
    "segment",
    "package",
]

# Permitir configurar la ubicación del modelo (útil tras mover carpetas)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "latest_model.pkl"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

app = FastAPI()


class InputData(BaseModel):
    region_id: float
    customer_type: str
    Y: float
    X: float
    size: float
    brand: str
    category: str
    sub_category: str
    segment: str
    package: str


@app.post("/predict/")
async def predict(data: InputData):
    """Recibir un registro, convertirlo en DataFrame y predecir."""
    df = pd.DataFrame([data.model_dump()])[SAFE_FEATURES]
    prediction = model.predict(df)
    proba = model.predict_proba(df)[:, 1]
    return {
        "prediction": prediction.tolist(),
        "probability": proba.tolist(),
    }
