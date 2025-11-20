from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

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

# Cargar el modelo entrenado
MODEL_PATH = Path("models") / "latest_model.pkl"
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
