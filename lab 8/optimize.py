import os
import pickle
import mlflow
import mlflow.sklearn
import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from optuna.visualization import plot_optimization_history, plot_param_importances

# Crear carpetas si no existen
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------------------
# Análisis y preprocesamiento mejorado
# -------------------------------
def load_and_analyze_data():
    """Cargar y analizar el dataset"""
    df = pd.read_csv("water_potability.csv")
    
    print("=" * 50)
    print("ANÁLISIS DEL DATASET")
    print("=" * 50)
    print("Distribución de clases:")
    print(df["Potability"].value_counts())
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    print(f"\nForma del dataset: {df.shape}")
    
    return df

def handle_missing_values(X):
    """Manejar valores nulos"""
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    
    print("\nValores nulos después del imputing:")
    print(X_imputed.isnull().sum())
    
    return X_imputed

def create_features(df):
    """Feature engineering"""
    df = df.copy()
    
    # Crear características de interacción
    if 'ph' in df.columns and 'Hardness' in df.columns:
        df['ph_hardness_ratio'] = df['ph'] / (df['Hardness'] + 1e-5)
    
    # Crear características polinomiales para columnas importantes
    for col in ['Solids', 'Chloramines', 'Sulfate']:
        if col in df.columns:
            df[f'{col}_squared'] = df[col] ** 2
    
    print(f"Características creadas. Nuevo shape: {df.shape}")
    return df

# -------------------------------
# Función objetivo mejorada con validación cruzada
# -------------------------------
def objective(trial):
    """Función objetivo mejorada con validación cruzada"""
    
    # Seleccionar tipo de modelo
    model_type = trial.suggest_categorical("model_type", ["xgboost", "random_forest"])
    
    if model_type == "xgboost":
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        }
        model = xgb.XGBClassifier(**params, random_state=42)
        

    # Usar validación cruzada para evaluación más robusta
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    
    with mlflow.start_run(nested=True):
        # Loggear todos los parámetros
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_type)
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train)):
            X_train_cv, X_val_cv = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_train_cv, y_train_cv)
            preds = model.predict(X_val_cv)
            f1 = f1_score(y_val_cv, preds)
            f1_scores.append(f1)
            
            # Loggear métricas por fold
            mlflow.log_metric(f"fold_{fold}_f1", f1)
        
        # Calcular métricas agregadas
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        cv_range = np.max(f1_scores) - np.min(f1_scores)
        
        # Loggear métricas agregadas
        mlflow.log_metric("mean_valid_f1", mean_f1)
        mlflow.log_metric("std_valid_f1", std_f1)
        mlflow.log_metric("cv_range", cv_range)
        mlflow.log_metric("min_f1", np.min(f1_scores))
        mlflow.log_metric("max_f1", np.max(f1_scores))
        
        # También evaluar en el conjunto de validación completo para comparación
        model.fit(X_train_scaled, y_train)
        valid_preds = model.predict(X_valid_scaled)
        valid_f1 = f1_score(y_valid, valid_preds)
        mlflow.log_metric("full_valid_f1", valid_f1)
        
        # Registrar el modelo
        input_example = X_valid_scaled[0].reshape(1, -1)
        mlflow.sklearn.log_model(model, "model", input_example=input_example)
        
        print(f"Trial {trial.number}: Mean F1 = {mean_f1:.4f} ± {std_f1:.4f}")
        
        return mean_f1

# -------------------------------
# Función para obtener el mejor modelo
# -------------------------------
def get_best_model(experiment_id):
    """Obtener el mejor modelo del experimento"""
    runs = mlflow.search_runs(experiment_id)
    if len(runs) == 0:
        raise ValueError("No se encontraron runs en el experimento")
    
    best_run = runs.sort_values("metrics.mean_valid_f1", ascending=False).iloc[0]
    best_model_id = best_run["run_id"]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")
    
    print(f"\nMejor modelo: Run {best_model_id}")
    print(f"Mejor F1-score: {best_run['metrics.mean_valid_f1']:.4f}")
    print(f"Tipo de modelo: {best_run['params.model_type']}")
    
    return best_model

# -------------------------------
# Evaluación final del modelo
# -------------------------------
def evaluate_final_model(model, X_test, y_test, scaler):
    """Evaluación completa del modelo final"""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    f1 = f1_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print("EVALUACIÓN FINAL DEL MODELO")
    print(f"{'='*50}")
    print(f"F1-Score en test: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return f1

# -------------------------------
# Función principal mejorada
# -------------------------------
def optimize_model():
    """Función principal de optimización mejorada"""
    
    # Crear experimento en MLflow
    experiment_name = "Optuna_XGBoost_Optimization_v2"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experimento creado: {experiment_name}")
    except Exception as e:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Usando experimento existente: {experiment_name}")

    mlflow.set_experiment(experiment_name)

    # Configurar estudio de Optuna
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner()
    )
    
    print(f"\nIniciando optimización con {N_TRIALS} trials...")
    print("Puedes monitorear el progreso con: mlflow ui")
    
    # Ejecutar optimización
    study.optimize(objective, n_trials=N_TRIALS, timeout=3600)  # 1 hora timeout

    # Resultados de la optimización
    print(f"\n{'='*50}")
    print("RESULTADOS DE LA OPTIMIZACIÓN")
    print(f"{'='*50}")
    print(f"Trials completados: {len(study.trials)}")
    print(f"Mejor F1-score: {study.best_value:.4f}")
    print("Mejores parámetros:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # Guardar gráficos
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_image("plots/optimization_history.png")
        
        fig2 = plot_param_importances(study)
        fig2.write_image("plots/param_importances.png")
        
        # Registrar gráficos en MLflow
        with mlflow.start_run(run_name="Optimization_Results", experiment_id=experiment_id):
            mlflow.log_artifact("plots/optimization_history.png", artifact_path="plots")
            mlflow.log_artifact("plots/param_importances.png", artifact_path="plots")
            mlflow.log_params(study.best_trial.params)
            mlflow.log_metric("best_f1_score", study.best_value)
    except Exception as e:
        print(f"Error al guardar gráficos: {e}")

    # Guardar el mejor modelo
    try:
        best_model = get_best_model(experiment_id)
        with open("models/best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        
        # Guardar el scaler también
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
            
        print("\nModelo y scaler guardados en la carpeta 'models/'")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")

    return study.best_value

# -------------------------------
# Configuración y ejecución
# -------------------------------
if __name__ == "__main__":
    # Configuración
    N_TRIALS = 50  # Reducido para prueba, puedes aumentarlo
    
    # Cargar y preparar datos
    print("Cargando y preparando datos...")
    df = load_and_analyze_data()
    
    # Separar características y objetivo
    X = df.drop(columns=["Potability"])
    y = df["Potability"]
    
    # Manejar valores nulos
    X = handle_missing_values(X)
    
    # Feature engineering
    X = create_features(X)
    
    # Dividir datos (70% train, 15% validation, 15% test)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
    
    print(f"\nDivision de datos:")
    print(f"Train: {X_train.shape[0]} muestras")
    print(f"Validación: {X_valid.shape[0]} muestras")
    print(f"Test: {X_test.shape[0]} muestras")
    
    # Escalado de características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    # Calcular class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    print(f"\nClass weights: {class_weights}")
    
    # Ejecutar optimización
    best_f1 = optimize_model()
    
    # Evaluar el mejor modelo en el conjunto de test
    try:
        with open("models/best_model.pkl", "rb") as f:
            best_model = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        test_f1 = evaluate_final_model(best_model, X_test, y_test, scaler)
        
        print(f"\n{'='*50}")
        if test_f1 >= 0.7:
            print("🎉 ¡Excelente resultado! F1-score > 0.7")
        elif test_f1 >= 0.6:
            print("✅ Buen resultado, pero puede mejorar")
        else:
            print("⚠️  Resultado bajo, considera revisar el dataset")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error en la evaluación final: {e}")
    
    print("\nOptimización completada. Ejecuta 'mlflow ui' para ver los resultados.")