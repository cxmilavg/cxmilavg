# Usa una imagen base de Python 3.10 slim
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /opt/airflow


# Instalar las dependencias del sistema
RUN apt-get update && apt-get install -y curl

# Instalar dependencias de Airflow y otras librerías
RUN pip install apache-airflow pandas


# Instalar las dependencias de Python
RUN pip install --upgrade pip


# Copiar los archivos del DAG al contenedor
COPY dags/ /opt/airflow/dags/

# Establecer las variables de entorno necesarias para Airflow
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__SQL_ALCHEMY_CONN=sqlite:////opt/airflow/airflow.db
ENV AIRFLOW__CORE__EXECUTOR=SequentialExecutor
ENV AIRFLOW__WEBSERVER__AUTHENTICATE=True
ENV AIRFLOW__WEBSERVER__AUTH_BACKEND=flask_appbuilder.security.manager.FAB_BASE_SECURITY_MANAGER


# Exponer el puerto en el que Airflow corre
EXPOSE 8080

# Comando para iniciar el servidor web de Airflow
CMD ["airflow", "webserver", "--port", "8080"]
