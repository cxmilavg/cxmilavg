from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear la aplicación FastAPI
app = FastAPI(
    title="API de Potabilidad del Agua",
    description="Esta API predice si una muestra de agua es potable o no usando un modelo XGBoost optimizado.",
    version="1.0.0"
)

# Variables globales para modelo y scaler
model = None
scaler = None

# Función para cargar el modelo
def load_model():
    global model, scaler
    try:
        # Ruta dentro del contenedor Docker
        model_path = "/app/models/best_model.pkl"
        scaler_path = "/app/models/scaler.pkl"
        
        # También probar rutas relativas para desarrollo local
        if not os.path.exists(model_path):
            model_path = "models/best_model.pkl"
        if not os.path.exists(scaler_path):
            scaler_path = "models/scaler.pkl"
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        logger.info("Modelo y scaler cargados exitosamente")
        logger.info(f"Tipo de modelo: {type(model)}")
        
    except FileNotFoundError as e:
        logger.error(f"No se encontraron los archivos del modelo: {e}")
        raise
    except Exception as e:
        logger.error(f"Error cargando el modelo: {e}")
        raise

# Cargar el modelo al iniciar la aplicación
@app.on_event("startup")
async def startup_event():
    load_model()

# Definir la estructura de entrada (Request Body)
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

# Ruta principal (GET)
@app.get("/")
def home():
    return {
        "mensaje": "API de Potabilidad del Agua",
        "estado": "Modelo cargado correctamente" if model is not None else "Modelo no cargado",
        "endpoints": {
            "documentación": "/docs",
            "salud": "/health",
            "predicción": "/predict"
        }
    }

# Ruta POST para predicciones
@app.post("/predict/")
def predict_potabilidad(data: WaterData):
    if model is None or scaler is None:
        return {"error": "Modelo no cargado correctamente"}
    
    try:
        input_data = np.array([[
            data.ph, data.Hardness, data.Solids, data.Chloramines,
            data.Sulfate, data.Conductivity, data.Organic_carbon,
            data.Trihalomethanes, data.Turbidity
        ]])

        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]
        prediction_proba = model.predict_proba(input_data_scaled)[0]

        return {
            "potabilidad": int(prediction),
            "probabilidad_no_potable": float(prediction_proba[0]),
            "probabilidad_potable": float(prediction_proba[1]),
            "clase_predicha": "Potable" if prediction == 1 else "No potable"
        }
    
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return {"error": f"Error en la predicción: {str(e)}"}

# Ruta para verificar salud del modelo
@app.get("/health/")
def health_check():
    if model is None or scaler is None:
        return {"status": "unhealthy", "error": "Modelo no cargado"}
    
    try:
        test_data = np.array([[7.0, 200, 20000, 5.0, 300, 500, 10, 80, 4.0]])
        test_data_scaled = scaler.transform(test_data)
        prediction = model.predict(test_data_scaled)
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "test_prediction": int(prediction[0])
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Ejecutar la aplicación
if __name__ == "__main__":
    import uvicorn
    # Cargar modelo antes de iniciar el servidor
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)