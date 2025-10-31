import os
import shutil
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

# Función para crear las carpetas
def create_folders(src_path=None, **kwargs):
    # Obtener la fecha de ejecución
    execution_date = kwargs.get('execution_date', datetime.now())
    folder_name = execution_date.strftime('%Y-%m-%d_%H-%M-%S')  # Usar la fecha como nombre de la carpeta

    # Crear la carpeta principal
    os.makedirs(folder_name)

    # Crear las subcarpetas
    os.makedirs(os.path.join(folder_name, 'raw'))
    os.makedirs(os.path.join(folder_name, 'splits'))
    os.makedirs(os.path.join(folder_name, 'models'))

    print(f"Carpetas creadas: {folder_name}, con subcarpetas 'raw', 'splits', 'models'.")

    # Verificar si el archivo 'data_1.csv' ya existe en la subcarpeta 'raw'
    raw_data_path = os.path.join(folder_name, 'raw', 'data_1.csv')

    if not os.path.exists(raw_data_path):
        if src_path:  # Si se proporciona una ruta de origen
            shutil.copy(src_path, raw_data_path)  # Copiar el archivo desde la ruta de origen
            print(f"Archivo 'data_1.csv' copiado desde {src_path} a {raw_data_path}")
        else:
            print(f"El archivo 'data_1.csv' no existe en la carpeta 'raw' y no se proporcionó una ruta de origen.")
    else:
        print(f"El archivo 'data_1.csv' ya existe en {raw_data_path}.")

    return folder_name

# Función para dividir los datos en entrenamiento y prueba
def split_data(folder_name):
    # Leer el archivo data_1.csv que se encuentra en la subcarpeta 'raw'
    raw_data_path = os.path.join(folder_name, 'raw', 'data_1.csv')
    data = pd.read_csv(raw_data_path)

    # Separar las variables X (características) y y (variable objetivo)
    X = data.drop(columns=['HiringDecision'])
    y = data['HiringDecision']

    # Aplicar el holdout para dividir en 80% entrenamiento y 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Guardar los conjuntos de entrenamiento y prueba en las subcarpetas 'splits'
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_csv(os.path.join(folder_name, 'splits', 'train.csv'), index=False)
    test_data.to_csv(os.path.join(folder_name, 'splits', 'test.csv'), index=False)

    print("Datos divididos y guardados en 'train.csv' y 'test.csv' en la carpeta 'splits'.")


def preprocess_and_train():
    # Leer los datasets de entrenamiento y prueba
    df_train = pd.read_csv(folder_name + '/splits/train.csv')
    df_test = pd.read_csv(folder_name + '/splits/test.csv')

    # Definir las características (X) y la variable objetivo (y)
    X_train = df_train.drop(columns=['HiringDecision'])
    y_train = df_train['HiringDecision']
    X_test = df_test.drop(columns=['HiringDecision'])
    y_test = df_test['HiringDecision']

    # Preprocesamiento de las variables numéricas y categóricas
    numeric_features = ['Age', 'ExperienceYears', 'InterviewScore', 'SkillScore', 'PersonalityScore']
    categorical_features = ['Gender', 'EducationLevel', 'PreviousCompanies', 'DistanceFromCompany', 'RecruitmentStrategy']

    # Definir el preprocesamiento para las variables numéricas
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Variante sin drop
    try:
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])
    except TypeError:
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])


    # Crear el ColumnTransformer para aplicar las transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Crear el pipeline con preprocesamiento y modelo RandomForest
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Entrenar el modelo
    pipeline.fit(X_train, y_train)

    # Guardar el modelo entrenado en un archivo joblib
    joblib.dump(pipeline, folder_name + '/models/hiring_model.joblib')

    # Realizar predicciones en el conjunto de prueba
    y_pred = pipeline.predict(X_test)

    # Evaluar el modelo con accuracy y f1-score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    # Imprimir los resultados
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score (positiva): {f1:.4f}')

def predict(file,model_path):

    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    predictions = pipeline.predict(input_data)
    print(f'La prediccion es: {predictions}')
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]

    return {'Predicción': labels[0]}

def gradio_interface():
    model_path = 'models/hiring_model.joblib'

    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Vale será contratada o no."
    )
    interface.launch(share=True)

# Crear las carpetas
folder_name = create_folders(src_path='data_1.csv')

# Dividir los datos en entrenamiento y prueba
split_data(folder_name)

# Preprocesar y entrenar el modelo
preprocess_and_train()
