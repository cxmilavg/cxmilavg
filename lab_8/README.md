Para importar las librerías se utilizará un entorno virtual.
Pasos para utilizarlo:

1. Desde la raíz repo (..\GitHub\cxmilavg)
3. Crear venv: "python -m venv .venv"
4. Activar venv: ".\.venv\Scripts\Activate.ps1"
5. Instalar las dependencias: pip install -r requirements.txt

¿Qué pasa si quiero agregar otra librería?
1. Instalar normalmente desde mi entorno virtual: pip install nombre_libreria
2. Luego actualiza el archivo requirements.txt para registrar todos los paquetes instalados: pip freeze > requirements.txt