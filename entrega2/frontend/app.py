import gradio as gr
import requests

# URL del backend FastAPI donde se realiza la predicción
backend_url = "http://backend:8000/predict/"

def get_prediction(feature1, feature2):
    # Realizar solicitud POST al backend
    response = requests.post(backend_url, json={"feature1": feature1, "feature2": feature2})
    if response.status_code == 200:
        return response.json()['prediction']
    else:
        return "Error: No se pudo obtener la predicción"

# Crear la interfaz Gradio
iface = gr.Interface(
    fn=get_prediction,
    inputs=[
        gr.inputs.Number(label="Feature 1"),
        gr.inputs.Number(label="Feature 2"),
    ],
    outputs=gr.outputs.Textbox(label="Predicción"),
    live=True,
)

iface.launch()
