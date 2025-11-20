import gradio as gr
import requests

# URL del backend FastAPI donde se realiza la predicción
backend_url = "http://backend:8000/predict/"


def get_prediction(feature1, feature2):
    """Solicitar una predicción al backend con las dos características."""
    response = requests.post(
        backend_url,
        json={"feature1": feature1, "feature2": feature2},
        timeout=10,
    )
    if response.status_code == 200:
        return response.json()["prediction"]
    return "Error: No se pudo obtener la predicción"


# Crear la interfaz Gradio usando el API actual (v4+)
iface = gr.Interface(
    fn=get_prediction,
    inputs=[gr.Number(label="Feature 1"), gr.Number(label="Feature 2")],
    outputs=gr.Textbox(label="Predicción"),
    live=True,
)

if __name__ == "__main__":
    iface.launch()
