# dags/dag_lineal.py
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import os
import requests

# Importar las funciones de tu archivo 'hiring_functions.py'
from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface

# Definir la fecha de inicio y la configuración del DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 1),
    'retries': 1,
    'catchup': False,
    'provide_context': True,
    'depends_on_past': False,
}

# Crear el DAG
dag = DAG(
    'hiring_lineal',  # dag_id único
    default_args=default_args,
    description='Pipeline de predicción de contratación',
    schedule_interval=None,  # Ejecución manual
)

# Tarea 1: Iniciar el pipeline
start_task = PythonOperator(
    task_id='start_pipeline',
    python_callable=lambda: print("Iniciando el pipeline..."),
    dag=dag,
)

# Tarea 2: Crear las carpetas correspondientes
create_folders_task = PythonOperator(
    task_id='create_folders',
    python_callable=create_folders,
    op_args=["/tmp/hiring_pipeline"],  # El path que quieras para la carpeta de ejecución
    provide_context=True,
    dag=dag,
)

# Tarea 3: Descargar el archivo 'data_1.csv'
def download_data(**kwargs):
    url = 'https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv'
    folder_name = kwargs['dag_run'].conf.get('folder_name')  # Se pasa el folder_name al DAG
    raw_data_path = os.path.join(folder_name, 'raw', 'data_1.csv')
    
    response = requests.get(url)
    
    # Guardar el archivo en la carpeta 'raw'
    with open(raw_data_path, 'wb') as f:
        f.write(response.content)
    print(f"Archivo 'data_1.csv' descargado en: {raw_data_path}")

download_data_task = PythonOperator(
    task_id='download_data',
    python_callable=download_data,
    provide_context=True,
    dag=dag,
)

# Tarea 4: Aplicar hold-out (split_data)
split_data_task = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    op_args=["/tmp/hiring_pipeline"],
    provide_context=True,
    dag=dag,
)

# Tarea 5: Preprocesar y entrenar el modelo
preprocess_and_train_task = PythonOperator(
    task_id='preprocess_and_train',
    python_callable=preprocess_and_train,
    op_args=["/tmp/hiring_pipeline"],
    provide_context=True,
    dag=dag,
)

# Tarea 6: Iniciar la interfaz de Gradio
gradio_interface_task = PythonOperator(
    task_id='gradio_interface',
    python_callable=gradio_interface,
    provide_context=True,
    dag=dag,
)

# Definir las dependencias entre las tareas
start_task >> create_folders_task >> download_data_task >> split_data_task >> preprocess_and_train_task >> gradio_interface_task
