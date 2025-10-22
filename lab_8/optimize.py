import os
import pickle
import mlflow
import mlflow.sklearn
import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# ===============================
# CONFIGURACIÓN MLFLOW
# ===============================
os.environ['MLFLOW_TRACKING_URI'] = 'file://' + os.path.join(os.getcwd(), 'mlruns')
mlflow.set_tracking_uri('file://' + os.path.join(os.getcwd(), 'mlruns'))

# Crear carpetas necesarias
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ===============================
# FUNCIONES AUXILIARES
# ===============================

def get_best_model(experiment_id):
    """Obtener el mejor modelo del experimento"""
    try:
        runs = mlflow.search_runs(experiment_id)
        if len(runs) == 0:
            raise ValueError("No se encontraron runs en el experimento")
        
        best_run = runs.sort_values("metrics.mean_valid_f1", ascending=False).iloc[0]
        best_model_id = best_run["run_id"]
        best_model = mlflow.sklearn.load_model(f"runs:/{best_model_id}/model")
        return best_model
    except Exception as e:
        print(f"Error cargando el mejor modelo: {e}")
        return None

def evaluate_final_model(model, X_test, y_test, scaler):
    """Evaluar el mejor modelo en el conjunto de test"""
    X_test_scaled = scaler.transform(X_test)
    test_preds = model.predict(X_test_scaled)
    test_f1 = f1_score(y_test, test_preds)
    
    print(f"\n{'='*50}")
    print("EVALUACIÓN FINAL DEL MODELO")
    print(f"{'='*50}")
    print(f"F1-Score en test: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_preds))
    
    return test_f1

# ===============================
# PREPROCESAMIENTO DE DATOS
# ===============================

def load_data():
    """Cargar el dataset"""
    return pd.read_csv("water_potability.csv")

def handle_missing_values(X):
    """Manejar valores nulos"""
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    return pd.DataFrame(X_imputed, columns=X.columns)

# ===============================
# FUNCIÓN OBJETIVO DE OPTUNA
# ===============================

def objective(trial):
    """Función objetivo para optimización con Optuna"""
    
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
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        }
        model = xgb.XGBClassifier(**params, random_state=42)
        
    else:  # random_forest
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": "balanced"
        }
        model = RandomForestClassifier(**params, random_state=42)

    # Validación cruzada
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []
    
    # Iniciar run de MLflow
    with mlflow.start_run(nested=True):
        # Loggear parámetros
        mlflow.log_param("model_type", model_type)
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Entrenar y evaluar con validación cruzada
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train)):
            X_train_cv, X_val_cv = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_train_cv, y_train_cv)
            preds = model.predict(X_val_cv)
            f1 = f1_score(y_val_cv, preds)
            f1_scores.append(f1)
            mlflow.log_metric(f"fold_{fold}_f1", f1)

        # Calcular métricas finales
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        
        mlflow.log_metric("mean_valid_f1", mean_f1)
        mlflow.log_metric("std_valid_f1", std_f1)
        
        # SOLUCIÓN A LOS WARNINGS: Configuración correcta de log_model
        # Crear input example para la firma del modelo
        input_example = X_train_scaled[:1].copy()
        
        # Inferir la firma automáticamente
        signature = mlflow.models.infer_signature(input_example, model.predict(input_example))
        
        # Guardar modelo sin warnings usando la API actualizada
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        print(f"Trial {trial.number}: F1 = {mean_f1:.4f} ± {std_f1:.4f}")
        return mean_f1

# ===============================
# FUNCIÓN PRINCIPAL DE OPTIMIZACIÓN
# ===============================

def optimize_model():
    """Función principal de optimización"""
    
    # Configurar experimento
    experiment_name = "Water_Potability_Optimization"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experimento creado: {experiment_name}")
    except Exception:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Usando experimento existente: {experiment_name}")

    mlflow.set_experiment(experiment_name)

    # Configurar estudio de Optuna
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print(f"Iniciando optimización con {N_TRIALS} trials...")
    
    # Ejecutar optimización
    study.optimize(objective, n_trials=N_TRIALS, timeout=3600)

    # Resultados
    print(f"\nOptimización completada")
    print(f"Trials completados: {len(study.trials)}")
    print(f"Mejor F1-score: {study.best_value:.4f}")
    print("Mejores parámetros:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # Guardar el mejor modelo
    try:
        best_model = get_best_model(experiment_id)
        if best_model is not None:
            with open("models/best_model.pkl", "wb") as f:
                pickle.dump(best_model, f)
            with open("models/scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            print("Modelo y scaler guardados exitosamente")
        else:
            print("No se pudo cargar el mejor modelo desde MLflow")
    except Exception as e:
        print(f"Error guardando el modelo: {e}")

    return study.best_value

# ===============================
# EJECUCIÓN PRINCIPAL
# ===============================

if __name__ == "__main__":
    # Configuración
    N_TRIALS = 50
    
    # Cargar y preparar datos
    print("Cargando y preparando datos...")
    df = load_data()
    
    # Separar características y objetivo
    X = df.drop(columns=["Potability"])
    y = df["Potability"]
    
    # Manejar valores nulos
    X = handle_missing_values(X)
    
    # Dividir datos
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    # Ejecutar optimización
    best_f1 = optimize_model()
    
    # Evaluar modelo final
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
            print("✅ Buen resultado")
        else:
            print("⚠️  Resultado bajo - considere revisar el enfoque")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error en evaluación final: {e}")
    
    print("\nProceso completado.")
    print("Para ver resultados ejecute: mlflow ui --backend-store-uri file:./mlruns")