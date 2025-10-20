from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# -----------------------------------------------------
# 1. Crear la aplicación FastAPI
# -----------------------------------------------------
app = FastAPI(
    title="API de Potabilidad del Agua",
    description="Esta API predice si una muestra de agua es potable o no usando un modelo XGBoost optimizado.",
    version="1.0.0"
)

# -----------------------------------------------------
# 2. Cargar el modelo entrenado
# -----------------------------------------------------
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------------------------------
# 3. Definir la estructura de entrada (Request Body)
# -----------------------------------------------------
class WaterData(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# -----------------------------------------------------
# 4. Ruta principal (GET)
# -----------------------------------------------------
@app.get("/")
def home():
    """
    Página de inicio con descripción básica del modelo y su propósito.
    """
    return {
        "mensaje": "Modelo de Clasificación de Potabilidad del Agua",
        "descripcion": (
            "Este modelo predice si una muestra de agua es potable (1) o no potable (0). "
            "Fue entrenado con datos de laboratorio usando XGBoost optimizado con Optuna."
        ),
        "entrada": {
            "ph": "pH del agua",
            "Hardness": "Dureza del agua",
            "Solids": "Sólidos disueltos totales (ppm)",
            "Chloramines": "Nivel de cloraminas (ppm)",
            "Sulfate": "Nivel de sulfato (ppm)",
            "Conductivity": "Conductividad eléctrica (μS/cm)",
            "Organic_carbon": "Carbono orgánico total (ppm)",
            "Trihalomethanes": "Nivel de trihalometanos (μg/L)",
            "Turbidity": "Turbidez (NTU)"
        },
        "salida": {"potabilidad": "1 si el agua es potable, 0 si no lo es"}
    }

# -----------------------------------------------------
# 5. Ruta POST para predicciones
# -----------------------------------------------------
@app.post("/potabilidad/")
def predict_potabilidad(data: WaterData):
    """
    Realiza una predicción sobre la potabilidad del agua.
    """
    # Convertir los datos de entrada a un array numpy para el modelo
    input_data = np.array([
        [
            data.ph,
            data.Hardness,
            data.Solids,
            data.Chloramines,
            data.Sulfate,
            data.Conductivity,
            data.Organic_carbon,
            data.Trihalomethanes,
            data.Turbidity
        ]
    ])

    # Realizar la predicción con el modelo cargado
    prediction = model.predict(input_data)[0]

    # Retornar la respuesta
    return {"potabilidad": int(prediction)}
