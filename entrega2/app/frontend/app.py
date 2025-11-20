import gradio as gr
import requests

# URL del backend FastAPI donde se realiza la predicción
backend_url = "http://backend:8000/predict/"


def get_prediction(
    region_id,
    customer_type,
    Y,
    X,
    size,
    brand,
    category,
    sub_category,
    segment,
    package,
):
    """Solicitar una predicción al backend con las mismas columnas usadas en el entrenamiento."""
    payload = {
        "region_id": region_id,
        "customer_type": customer_type,
        "Y": Y,
        "X": X,
        "size": size,
        "brand": brand,
        "category": category,
        "sub_category": sub_category,
        "segment": segment,
        "package": package,
    }
    try:
        response = requests.post(backend_url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        prob = data["probability"][0]
        pred = data["prediction"][0]
        return f"Predicción: {pred} | Probabilidad compra: {prob:.3f}"
    except Exception as exc:
        return f"Error al obtener la predicción: {exc}"


input_components = [
    gr.Number(label="Region ID"),
    gr.Textbox(label="Tipo de cliente"),
    gr.Number(label="Coordenada Y"),
    gr.Number(label="Coordenada X"),
    gr.Number(label="Tamaño"),
    gr.Textbox(label="Marca"),
    gr.Textbox(label="Categoría"),
    gr.Textbox(label="Subcategoría"),
    gr.Textbox(label="Segmento"),
    gr.Textbox(label="Paquete"),
]

iface = gr.Interface(
    fn=get_prediction,
    inputs=input_components,
    outputs=gr.Textbox(label="Respuesta"),
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
