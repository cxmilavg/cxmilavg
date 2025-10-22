# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos del proyecto al contenedor
COPY . /app

# Crear un entorno virtual dentro del contenedor
RUN python3 -m venv /app/venv

# Activar el entorno virtual y actualizar pip
RUN /app/venv/bin/pip install --upgrade pip

# Instalar las dependencias desde requirements.txt dentro del entorno virtual
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Exponer el puerto 8000 para que la API sea accesible
EXPOSE 8000

# Configurar la variable de entorno para activar el entorno virtual cuando se ejecute el contenedor
ENV PATH="/app/venv/bin:$PATH"

# Comando para ejecutar la aplicación con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
