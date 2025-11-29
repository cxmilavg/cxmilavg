FROM python:3.9-slim

ENV AIRFLOW_HOME=/opt/airflow
WORKDIR ${AIRFLOW_HOME}

# Dependencias de sistema necesarias para pandas/sklearn/optuna/etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        libffi-dev \
        libblas-dev \
        liblapack-dev \
        git \
        curl \
        libre2-dev \
        libabsl-dev \
        cmake && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "google-re2==1.1.20240702" && \
    pip install --no-cache-dir -r requirements.txt


# Copiamos DAGs y datos base
COPY dags dags
COPY data data

# Aseguramos carpetas que el pipeline usa
RUN mkdir -p models data/raw data/processed data/predictions logs plugins

EXPOSE 8080

# airflow standalone inicializa DB y crea usuario admin automáticamente
CMD ["airflow", "standalone"]
