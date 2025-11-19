from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import mlflow
from pathlib import Path

# Cargar el modelo entrenado (desde el directorio donde se guarda el modelo)
MODEL_PATH = Path("/path/to/your/model") / "latest_model.pkl"  # Actualiza con la ruta exacta
model = joblib.load(MODEL_PATH)

app = FastAPI()

class InputData(BaseModel):
    feature1: float
    feature2: float

@app.post("/predict/")
async def predict(data: InputData):
    input_data = [[data.feature1, data.feature2]]  # Adaptar según los datos recibidos
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}
