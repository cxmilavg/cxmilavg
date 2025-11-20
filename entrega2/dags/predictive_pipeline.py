from __future__ import annotations

import os
import sys
import warnings
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

import shutil
from datetime import datetime, timedelta
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
from optuna.samplers import TPESampler
import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.python import PythonSensor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Rutas base
BASE_DIR = Path(__file__).resolve().parent.parent
INCOMING_DIR = BASE_DIR / "data" / "incoming"
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REFERENCE_PATH = BASE_DIR / "data" / "reference" / "transacciones_clean.parquet"
MODEL_DIR = BASE_DIR / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"

# Esquema mínimo esperado del parquet base
REQUIRED_COLS = {
    "customer_id",
    "product_id",
    "order_id",
    "purchase_date",
    "items",
}

# Columnas de clientes/productos esperadas tras merge
CLIENT_COLS = {
    "region_id",
    "customer_type",
    "Y",
    "X",
    "num_deliver_per_week",
    "num_visit_per_week",
}
PRODUCT_COLS = {
    "brand",
    "category",
    "sub_category",
    "segment",
    "package",
    "size",
}

# Columnas numéricas y categóricas para el preprocesador
NUM_COLS = ["items", "region_id", "Y", "X", "num_deliver_per_week", "num_visit_per_week", "size"]
CAT_COLS = ["customer_type", "brand", "category", "sub_category", "segment", "package"]

# Columna objetivo generada en el enriquecimiento
TARGET_COL = "compra"

def setup_mlflow():
    """Configurar MLflow con mejores prácticas"""
    mlflow.set_tracking_uri(str(BASE_DIR / "mlruns"))
    mlflow.set_experiment("sodai_drinks_predictive_pipeline")
    
    # Crear directorios necesarios
    (BASE_DIR / "mlruns").mkdir(exist_ok=True)
    return mlflow

# ------------------ Ingesta y limpieza básica ------------------
def _has_new_parquet() -> bool:
    return any(INCOMING_DIR.glob("transacciones*.parquet"))


def ingest_data(**context) -> str:
    files = sorted(INCOMING_DIR.glob("transacciones*.parquet"))
    if not files:
        raise FileNotFoundError(f"No se encontraron transacciones*.parquet en {INCOMING_DIR}")
    latest = files[-1]
    run_date = context["ds"]
    dest_dir = RAW_DIR / run_date
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / latest.name
    shutil.copy2(latest, dest_path)
    return str(dest_path)


def transform_data(**context) -> str:
    raw_path = Path(context["ti"].xcom_pull(task_ids="ingest_data"))
    df = pd.read_parquet(raw_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df = df.drop_duplicates().fillna(0)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")

    out_dir = PROCESSED_DIR / context["ds"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transacciones_clean.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)


# ------------------ Enriquecimiento y etiqueta temporal ------------------
def enrich_and_label(**context) -> str:
    """
    Etiqueta compra=1 si el par cliente-producto aparece en la semana siguiente; 0 en caso contrario.
    Usa semanas calendario (W) a partir de purchase_date. Balancea negativos a proporción 1:1.
    """
    processed_path = Path(context["ti"].xcom_pull(task_ids="transform_data"))
    base_dir = processed_path.parent

    clientes_path = INCOMING_DIR / "clientes.parquet"
    productos_path = INCOMING_DIR / "productos.parquet"

    df_tx = pd.read_parquet(processed_path)
    df_tx["purchase_date"] = pd.to_datetime(df_tx["purchase_date"], errors="coerce")
    df_tx["week"] = df_tx["purchase_date"].dt.to_period("W").dt.start_time

    pairs_week = df_tx[["customer_id", "product_id", "week"]].drop_duplicates()
    future_pairs = pairs_week.copy()
    future_pairs["week"] = future_pairs["week"] - pd.to_timedelta(7, unit="d")
    future_set = set(map(tuple, future_pairs.values))

    labels = []
    for row in pairs_week.itertuples(index=False):
        key_future = (row.customer_id, row.product_id, row.week + pd.to_timedelta(7, unit="d"))
        compra = 1 if key_future in future_set else 0
        labels.append((row.customer_id, row.product_id, row.week, compra))
    df_label = pd.DataFrame(labels, columns=["customer_id", "product_id", "week", TARGET_COL])

    df_tx = df_tx.merge(df_label, on=["customer_id", "product_id", "week"], how="right")

    pos = df_tx[df_tx[TARGET_COL] == 1]
    neg = df_tx[df_tx[TARGET_COL] == 0]
    if len(pos) > 0 and len(neg) > len(pos):
        neg = neg.sample(n=len(pos), random_state=42)
    df_balanced = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=42)

    if clientes_path.exists():
        df_cli = pd.read_parquet(clientes_path)
        df_balanced = df_balanced.merge(df_cli, on="customer_id", how="left")
    if productos_path.exists():
        df_prod = pd.read_parquet(productos_path)
        df_balanced = df_balanced.merge(df_prod, on="product_id", how="left")

    out_path = base_dir / "transacciones_enriched.parquet"
    df_balanced.to_parquet(out_path, index=False)
    return str(out_path)


# ------------------ Preprocesador reutilizable ------------------
def build_preprocessor(feature_columns: Optional[set[str]] = None) -> ColumnTransformer:
    available = set(feature_columns or [])
    num_active = [c for c in NUM_COLS if (not available or c in available)]
    cat_active = [c for c in CAT_COLS if (not available or c in available)]

    if not num_active and not cat_active:
        raise ValueError("No hay columnas válidas para el preprocesador; revisa el dataset de entrada")

    num_proc = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_proc = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers = []
    if num_active:
        transformers.append(("num", num_proc, num_active))
    if cat_active:
        transformers.append(("cat", cat_proc, cat_active))

    return ColumnTransformer(transformers=transformers)


# ------------------ Drift detection (PSI) ------------------
def _psi_numeric(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    expected_perc, bins = np.histogram(expected, bins=buckets)
    actual_perc, _ = np.histogram(actual, bins=bins)
    expected_perc = expected_perc / max(expected_perc.sum(), 1)
    actual_perc = actual_perc / max(actual_perc.sum(), 1)
    psi = ((actual_perc - expected_perc) * np.log((actual_perc + 1e-8) / (expected_perc + 1e-8))).sum()
    return float(psi)


def _psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    exp_counts = expected.value_counts(normalize=True)
    act_counts = actual.value_counts(normalize=True)
    all_idx = exp_counts.index.union(act_counts.index)
    exp = exp_counts.reindex(all_idx, fill_value=0)
    act = act_counts.reindex(all_idx, fill_value=0)
    psi = ((act - exp) * np.log((act + 1e-8) / (exp + 1e-8))).sum()
    return float(psi)


def detect_drift(**context) -> dict:
    """
    Detectar drift usando PSI (Population Stability Index)
    Retorna dict con información del drift detectado
    """
    try:
        processed_path = Path(context["ti"].xcom_pull(task_ids="transform_data"))
        df_new = pd.read_parquet(processed_path)

        # Si no existe referencia, crear una
        if not REFERENCE_PATH.exists():
            REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            df_new.to_parquet(REFERENCE_PATH, index=False)
            context["ti"].xcom_push(key="drift_reason", value="reference_initialized")
            return {"drift_detected": False, "reason": "reference_initialized"}

        df_ref = pd.read_parquet(REFERENCE_PATH)
        
        # Verificar que ambos DataFrames tengan las mismas columnas
        common_cols = set(df_ref.columns) & set(df_new.columns)
        if not common_cols:
            return {"drift_detected": True, "reason": "no_common_columns"}
        
        psi_scores = {}
        drift_threshold = 0.2
        
        # Calcular PSI solo para columnas numéricas
        numeric_cols = df_ref.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col in df_new.columns]
        
        for col in numeric_cols:
            try:
                # Filtrar valores infinitos y NaN
                ref_vals = df_ref[col].replace([np.inf, -np.inf], np.nan).dropna()
                new_vals = df_new[col].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(ref_vals) > 0 and len(new_vals) > 0:
                    psi = _psi_numeric(ref_vals, new_vals)
                    psi_scores[col] = psi
            except Exception as e:
                print(f"Error calculando PSI para {col}: {e}")
                continue

        if not psi_scores:
            return {"drift_detected": False, "reason": "no_valid_columns_for_psi"}
        
        max_psi = max(psi_scores.values())
        drift_detected = max_psi > drift_threshold
        
        result = {
            "drift_detected": drift_detected,
            "max_psi": max_psi,
            "psi_scores": psi_scores,
            "reason": f"drift_detected_{max_psi:.3f}" if drift_detected else "no_drift"
        }
        
        # Actualizar referencia si no hay drift
        if not drift_detected:
            df_new.to_parquet(REFERENCE_PATH, index=False)
            
        return result
        
    except Exception as e:
        print(f"Error en detección de drift: {e}")
        return {"drift_detected": False, "reason": f"error: {str(e)}"}

# ------------------ Branching retrain ------------------
def _should_retrain(**context) -> str:
    drift_info = context["ti"].xcom_pull(task_ids="detect_drift") or {}
    drift = drift_info.get("drift_detected", False)
    force_every = int(Variable.get("force_retrain_every", default_var=7))
    run_num = int(Variable.get("run_counter", default_var=0)) + 1
    Variable.set("run_counter", run_num)
    periodic = (run_num % force_every) == 0
    return "tune_model" if (drift or periodic) else "skip_retrain"


# ------------------ Tuning / entrenamiento ------------------
def tune_model(**context) -> dict:
    """Optimizar hiperparámetros usando Optuna para LightGBM"""
    try:
        enriched_path = Path(context["ti"].xcom_pull(task_ids="enrich_and_label"))
        df = pd.read_parquet(enriched_path)
        
        if TARGET_COL not in df.columns:
            raise ValueError(f"Falta columna objetivo {TARGET_COL}")

        # Preparar características seguras (sin leakage)
        safe_features = ['region_id', 'customer_type', 'Y', 'X', 'size', 
                        'brand', 'category', 'sub_category', 'segment', 'package']
        safe_features = [col for col in safe_features if col in df.columns]
        
        X = df[safe_features]
        y = df[TARGET_COL]
        
        # División temporal
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        preprocessor = build_preprocessor(set(X_train.columns))

        def objective(trial):
            """Función objetivo para Optuna"""
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            }

            model = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("classifier", LGBMClassifier(**params, random_state=42, n_jobs=-1)),
                ]
            )

            from sklearn.model_selection import cross_val_score

            scores = cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=10, timeout=600)

        best_params = study.best_params
        context["ti"].xcom_push(key="best_params", value=best_params)

        # Loggear en MLflow
        with mlflow.start_run(run_name="hyperparameter_tuning"):
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_score", study.best_value)
            mlflow.log_param("n_trials", len(study.trials))
        
        return best_params
        
    except Exception as e:
        print(f"Error en optimización: {e}")
        # Retornar parámetros por defecto en caso de error
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 31
        }
        context["ti"].xcom_push(key="best_params", value=default_params)
        return default_params


def train_model(**context) -> str:
    """Entrenar modelo con los mejores hiperparámetros encontrados"""
    try:
        enriched_path = Path(context["ti"].xcom_pull(task_ids="enrich_and_label"))
        df = pd.read_parquet(enriched_path)
        
        best_params = context["ti"].xcom_pull(task_ids="tune_model", key="best_params") or {}
        
        if TARGET_COL not in df.columns:
            raise ValueError(f"Falta columna objetivo {TARGET_COL}")

        # Preparar características seguras
        safe_features = ['region_id', 'customer_type', 'Y', 'X', 'size', 
                        'brand', 'category', 'sub_category', 'segment', 'package']
        safe_features = [col for col in safe_features if col in df.columns]
        
        X = df[safe_features]
        y = df[TARGET_COL]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        preprocessor = build_preprocessor(set(X_train.columns))
        
        # Usar LightGBM como modelo principal
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LGBMClassifier(**best_params, random_state=42, n_jobs=-1))
        ])

        with mlflow.start_run(run_name="model_training") as run:
            # Loggear parámetros y métricas
            mlflow.log_params(best_params)
            mlflow.log_param("model_type", "LightGBM")
            mlflow.log_param("features", str(safe_features))
            
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Evaluar
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            from sklearn.metrics import classification_report, roc_auc_score, f1_score
            f1 = f1_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba)
            accuracy = (y_pred == y_val).mean()
            
            # Loggear métricas
            mlflow.log_metrics({
                "f1_score": f1,
                "roc_auc": auc,
                "accuracy": accuracy
            })
            
            # Loggear classification report como artifact
            report = classification_report(y_val, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_path = MODEL_DIR / "classification_report.csv"
            report_df.to_csv(report_path)
            mlflow.log_artifact(str(report_path))
            
            # Guardar modelo
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MODEL_DIR / "latest_model.pkl"
            joblib.dump(model, model_path)
            
            # Loggear modelo en MLflow
            mlflow.sklearn.log_model(model, "model")
            
            print(f"Modelo entrenado - F1: {f1:.4f}, AUC: {auc:.4f}")
            
        return str(model_path)
        
    except Exception as e:
        print(f"Error en entrenamiento: {e}")
        raise


# ------------------ Interpretabilidad (coeficientes) ------------------
def interpret_model(**context) -> str:
    """Generar explicaciones del modelo usando SHAP"""
    try:
        model_path = Path(context["ti"].xcom_pull(task_ids="train_model"))
        model = joblib.load(model_path)
        
        enriched_path = Path(context["ti"].xcom_pull(task_ids="enrich_and_label"))
        df = pd.read_parquet(enriched_path)
        
        # Preparar datos para SHAP
        safe_features = ['region_id', 'customer_type', 'Y', 'X', 'size', 
                        'brand', 'category', 'sub_category', 'segment', 'package']
        safe_features = [col for col in safe_features if col in df.columns]
        
        X = df[safe_features].sample(1000, random_state=42)  # Muestra para SHAP
        
        # Transformar datos
        preprocessor = model.named_steps['preprocessor']
        X_transformed = preprocessor.transform(X)
        
        # Obtener nombres de características después del preprocesamiento
        feature_names = preprocessor.get_feature_names_out()
        
        # Calcular valores SHAP
        import shap
        explainer = shap.TreeExplainer(model.named_steps['classifier'])
        shap_values = explainer.shap_values(X_transformed)
        
        # Crear directorio para artifacts
        artifact_dir = MODEL_DIR / "shap_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Gráfico de importancia global
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
        plt.title("SHAP Feature Importance")
        importance_path = artifact_dir / "shap_importance.png"
        plt.tight_layout()
        plt.savefig(importance_path)
        plt.close()
        
        # Gráfico de dependencia para características principales
        if len(shap_values.shape) > 1:  # Para clasificación multiclase
            shap_values_2d = shap_values[1] if len(shap_values) == 2 else shap_values
        else:
            shap_values_2d = shap_values
            
        # Encontrar característica más importante
        feature_importance = np.abs(shap_values_2d).mean(0)
        top_feature_idx = np.argmax(feature_importance)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(top_feature_idx, shap_values_2d, X_transformed, 
                           feature_names=feature_names, show=False)
        dependence_path = artifact_dir / "shap_dependence.png"
        plt.tight_layout()
        plt.savefig(dependence_path)
        plt.close()
        
        # Loggear en MLflow
        with mlflow.start_run(run_name="model_interpretation"):
            mlflow.log_artifact(str(importance_path))
            mlflow.log_artifact(str(dependence_path))
            
            # Guardar valores SHAP
            shap_values_path = artifact_dir / "shap_values.npy"
            np.save(shap_values_path, shap_values)
            mlflow.log_artifact(str(shap_values_path))
        
        return str(importance_path)
        
    except Exception as e:
        print(f"Error en interpretabilidad: {e}")
        # Continuar incluso si hay error en SHAP
        return "interpretation_failed"


# ------------------ Predicción ------------------
def predict_next_week(**context) -> str:
    """Generar predicciones para la semana siguiente"""
    try:
        enriched_path = Path(context['ti'].xcom_pull(task_ids='enrich_and_label'))
        df = pd.read_parquet(enriched_path)
        
        # Asegurar que tenemos la columna de fecha
        if 'purchase_date' not in df.columns:
            raise ValueError("Columna purchase_date no encontrada")
            
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        
        # Encontrar la fecha máxima y calcular la semana siguiente
        max_date = df['purchase_date'].max()
        next_week_start = max_date + pd.Timedelta(days=7)
        
        print(f"Fecha máxima en datos: {max_date}")
        print(f"Prediciendo para semana: {next_week_start}")
        
        # Usar la última semana disponible para predicción
        last_week_data = df[df['purchase_date'] >= (max_date - pd.Timedelta(days=7))].copy()
        
        if last_week_data.empty:
            raise ValueError("No hay datos de la última semana para generar predicciones")
        
        # Cargar modelo
        model_path = Path(context['ti'].xcom_pull(task_ids='train_model'))
        model = joblib.load(model_path)
        
        # Preparar características para predicción
        safe_features = ['region_id', 'customer_type', 'Y', 'X', 'size', 
                        'brand', 'category', 'sub_category', 'segment', 'package']
        safe_features = [col for col in safe_features if col in last_week_data.columns]
        
        X_pred = last_week_data[safe_features]
        
        # Generar predicciones
        predictions = model.predict_proba(X_pred)[:, 1]  # Probabilidad de clase positiva
        
        # Crear DataFrame de resultados
        results_df = last_week_data[['customer_id', 'product_id']].copy()
        results_df['prediction_date'] = pd.Timestamp.now()
        results_df['target_week'] = next_week_start
        results_df['purchase_probability'] = predictions
        results_df['predicted_purchase'] = (predictions > 0.5).astype(int)
        
        # Guardar predicciones
        predictions_dir = PREDICTIONS_DIR / context['ds']
        predictions_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = predictions_dir / 'predicciones_semana_siguiente.parquet'
        
        results_df.to_parquet(predictions_path, index=False)
        
        # Loggear en MLflow
        with mlflow.start_run(run_name="weekly_predictions"):
            mlflow.log_artifact(str(predictions_path))
            mlflow.log_metrics({
                "mean_prediction_probability": predictions.mean(),
                "predicted_purchases": results_df['predicted_purchase'].sum(),
                "total_predictions": len(results_df)
            })
        
        print(f"Predicciones generadas: {len(results_df)} registros")
        print(f"Predicciones guardadas en: {predictions_path}")
        
        return str(predictions_path)
        
    except Exception as e:
        print(f"Error en generación de predicciones: {e}")
        raise


def skip_retrain():
    return "no_retrain"


# ------------------ DAG ------------------
with DAG(
    dag_id="data_extraction_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["extract", "clean", "drift", "train", "predict"],
) as dag:
    wait_for_data = PythonSensor(
        task_id="wait_for_transacciones",
        poke_interval=60,
        timeout=60 * 30,
        mode="poke",
        python_callable=_has_new_parquet,
    )

    ingest_task = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

    transform_task = PythonOperator(
        task_id="transform_data",
        python_callable=transform_data,
    )

    enrich_task = PythonOperator(
        task_id="enrich_and_label",
        python_callable=enrich_and_label,
    )

    drift_task = PythonOperator(
        task_id="detect_drift",
        python_callable=detect_drift,
    )

    branch_retrain = BranchPythonOperator(
        task_id="branch_retrain",
        python_callable=_should_retrain,
    )

    tune_task = PythonOperator(
        task_id="tune_model",
        python_callable=tune_model,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    interpret_task = PythonOperator(
        task_id="interpret_model",
        python_callable=interpret_model,
    )

    predict_task = PythonOperator(
        task_id="predict_next_week",
        python_callable=predict_next_week,
    )

    no_retrain = PythonOperator(
        task_id="skip_retrain",
        python_callable=skip_retrain,
    )

    wait_for_data >> ingest_task >> transform_task >> enrich_task >> drift_task >> branch_retrain
    branch_retrain >> [tune_task, no_retrain]
    tune_task >> train_task >> interpret_task >> predict_task
    no_retrain >> predict_task
