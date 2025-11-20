FROM python:3.9-slim

ENV AIRFLOW_HOME=/opt/airflow
WORKDIR ${AIRFLOW_HOME}

# Dependencias de sistema para compilar paquetes (pandas, scikit-learn, google-re2, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        default-libmysqlclient-dev \
        git \
        libffi-dev \
        libssl-dev \
        libblas-dev \
        liblapack-dev \
        curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiamos el código del proyecto
COPY dags dags
COPY data data

# Carpetas consumidas por el pipeline si no existen
RUN mkdir -p models data/raw data/processed data/predictions

# Airflow genera la base y un usuario admin por defecto con 'airflow standalone'
EXPOSE 8080
CMD ["airflow", "standalone"]
