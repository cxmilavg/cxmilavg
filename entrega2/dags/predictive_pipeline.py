from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
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

mlflow.set_tracking_uri(str(BASE_DIR / "mlruns"))
mlflow.set_experiment("predictive_pipeline")


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
def build_preprocessor() -> ColumnTransformer:
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
    return ColumnTransformer(
        transformers=[
            ("num", num_proc, NUM_COLS),
            ("cat", cat_proc, CAT_COLS),
        ]
    )


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
    processed_path = Path(context["ti"].xcom_pull(task_ids="transform_data"))
    df_new = pd.read_parquet(processed_path)

    if not REFERENCE_PATH.exists():
        REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(processed_path, REFERENCE_PATH)
        return {"drift_detected": False, "reason": "reference_initialized"}

    df_ref = pd.read_parquet(REFERENCE_PATH)

    psi_scores = {}
    for col in NUM_COLS:
        if col in df_ref.columns and col in df_new.columns:
            psi_scores[col] = _psi_numeric(df_ref[col], df_new[col])
    for col in CAT_COLS:
        if col in df_ref.columns and col in df_new.columns:
            psi_scores[col] = _psi_categorical(df_ref[col], df_new[col])

    max_psi = max(psi_scores.values()) if psi_scores else 0.0
    drift = max_psi > 0.2
    return {"drift_detected": drift, "max_psi": max_psi, "psi_scores": psi_scores}


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
    enriched_path = Path(context["ti"].xcom_pull(task_ids="enrich_and_label"))
    df = pd.read_parquet(enriched_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Falta columna objetivo {TARGET_COL} para tuning/entrenamiento")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = build_preprocessor()

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 10, log=True)
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=200, C=C, n_jobs=-1)),
            ]
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return f1_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, timeout=300)
    best_params = study.best_params
    context["ti"].xcom_push(key="best_params", value=best_params)
    return best_params


def train_model(**context) -> str:
    enriched_path = Path(context["ti"].xcom_pull(task_ids="enrich_and_label"))
    df = pd.read_parquet(enriched_path)
    best_params = context["ti"].xcom_pull(task_ids="tune_model", key="best_params") or {}
    if TARGET_COL not in df.columns:
        raise ValueError(f"Falta columna objetivo {TARGET_COL} para entrenamiento")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = build_preprocessor()
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=200, **best_params)),
        ]
    )

    with mlflow.start_run(run_name="retrain"):
        mlflow.log_params(best_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds)
        auc = roc_auc_score(y_val, preds)
        mlflow.log_metric("f1_val", f1)
        mlflow.log_metric("roc_auc_val", auc)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "latest_model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))
    return str(model_path)


# ------------------ Interpretabilidad (coeficientes) ------------------
def interpret_model(**context) -> str:
    model_path = Path(context["ti"].xcom_pull(task_ids="train_model"))
    model = joblib.load(model_path)
    clf = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    coefs = getattr(clf, "coef_", None)

    artifact_dir = MODEL_DIR / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    coef_txt = artifact_dir / "coefficients.txt"
    coef_png = artifact_dir / "feature_importance.png"

    if coefs is not None:
        np.savetxt(coef_txt, coefs)
        coef_series = pd.Series(coefs.flatten(), index=feature_names)
        top = coef_series.abs().sort_values(ascending=False).head(20)
        plt.figure(figsize=(8, 6))
        top.sort_values().plot(kind="barh")
        plt.tight_layout()
        plt.savefig(coef_png)
        plt.close()
        mlflow.log_artifact(str(coef_png))
        mlflow.log_artifact(str(coef_txt))
    return str(coef_png)


# ------------------ Predicción ------------------
def predict_next_week(**context) -> str:
    """
    Genera predicciones para la semana siguiente a la mas reciente (t_max+7d).
    Usa las observaciones de la ultima semana como base de features.
    """
    enriched_path = Path(context['ti'].xcom_pull(task_ids='enrich_and_label'))
    df = pd.read_parquet(enriched_path)
    if 'week' not in df.columns:
        df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')
        df['week'] = df['purchase_date'].dt.to_period('W').dt.start_time

    t_max = df['week'].max()
    week_pred = t_max + pd.to_timedelta(7, unit='d')

    df_latest = df[df['week'] == t_max].copy()
    if TARGET_COL in df_latest.columns:
        df_latest = df_latest.drop(columns=[TARGET_COL])
    X = df_latest.drop(columns=['week'], errors='ignore')

    model_path = Path(context['ti'].xcom_pull(task_ids='train_model'))
    model = joblib.load(model_path)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)

    df_out = df_latest[['customer_id', 'product_id']].copy()
    df_out['predict_week'] = week_pred
    df_out['score'] = proba

    out_dir = PREDICTIONS_DIR / context['ds']
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'predicciones.parquet'
    df_out.to_parquet(out_path, index=False)
    mlflow.log_artifact(str(out_path))
    return str(out_path)


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
