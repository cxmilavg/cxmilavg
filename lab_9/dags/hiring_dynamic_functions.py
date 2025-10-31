# dags/hiring_dynamic_functions.py

import os
import glob
import shutil
import joblib
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


# -------------------------- utilidades internas -------------------------- #
def _resolve_folder_name(folder_name=None, **kwargs) -> str:
    if folder_name:
        return folder_name
    if kwargs and 'folder_name' in kwargs and kwargs['folder_name']:
        return kwargs['folder_name']
    execution_date = kwargs.get('execution_date', datetime.now())
    return execution_date.strftime('%Y-%m-%d_%H-%M-%S')


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _infer_columns(df: pd.DataFrame, target_col: str):
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    cat_cols = df.columns.difference(num_cols + [target_col]).tolist()
    return num_cols, cat_cols


def _copy_if_needed(src_path: str | None, dst_path: str, label: str):
    """
    Copia un archivo si se proporciona src_path y el destino no existe.
    Imprime mensajes claros para facilitar el debug en Airflow.
    """
    if os.path.exists(dst_path):
        print(f"[create_folders] '{label}' ya existe en: {dst_path}")
        return

    if src_path:
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f"[create_folders] '{label}' copiado desde {src_path} a {dst_path}")
        else:
            print(f"[create_folders][AVISO] Ruta de origen no existe para '{label}': {src_path}")
    else:
        print(f"[create_folders][AVISO] No se proporcionó ruta de origen para '{label}'. "
              f"Si lo necesitas, pasa src_path correspondiente.")


# -------------------------------- (1) create_folders -------------------------------- #
def create_folders(
    folder_name: str | None = None,
    src_path1: str | None = None,  # ruta para data_1.csv
    src_path2: str | None = None,  # ruta para data_2.csv (opcional)
    **kwargs
) -> str:
    """
    Crea carpeta de ejecución y subcarpetas:
      - raw
      - preprocessed
      - splits
      - models
    Además, copia:
      - data_1.csv desde src_path1 -> raw/data_1.csv (si se provee/exists)
      - data_2.csv desde src_path2 -> raw/data_2.csv (opcional)
    Retorna la ruta de la carpeta de ejecución.
    """
    run_folder = _resolve_folder_name(folder_name, **kwargs)

    _ensure_dir(run_folder)
    raw_dir = os.path.join(run_folder, 'raw')
    _ensure_dir(raw_dir)
    _ensure_dir(os.path.join(run_folder, 'preprocessed'))
    _ensure_dir(os.path.join(run_folder, 'splits'))
    _ensure_dir(os.path.join(run_folder, 'models'))

    print(f"[create_folders] Estructura creada en: {run_folder}")

    # Copiar data_1.csv y data_2.csv si corresponde
    _copy_if_needed(src_path1, os.path.join(raw_dir, 'data_1.csv'), 'data_1.csv')
    _copy_if_needed(src_path2, os.path.join(raw_dir, 'data_2.csv'), 'data_2.csv')

    return run_folder


# -------------------------------- (2) load_ands_merge -------------------------------- #
def load_ands_merge(folder_name: str | None = None, **kwargs) -> str:
    run_folder = _resolve_folder_name(folder_name, **kwargs)
    raw_dir = os.path.join(run_folder, 'raw')
    prep_dir = os.path.join(run_folder, 'preprocessed')
    _ensure_dir(prep_dir)

    p1 = os.path.join(raw_dir, 'data_1.csv')
    p2 = os.path.join(raw_dir, 'data_2.csv')

    paths = []
    if os.path.exists(p1):
        paths.append(p1)
    if os.path.exists(p2):
        paths.append(p2)

    if not paths:
        raise FileNotFoundError(
            f"[load_ands_merge] No se encontró ninguno de: {p1} ni {p2}. "
            "Asegura pasar src_path1/src_path2 en create_folders o colocar los archivos en raw."
        )

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)

    out_path = os.path.join(prep_dir, 'data_merged.csv')
    df.to_csv(out_path, index=False)
    print(f"[load_ands_merge] Guardado {out_path} (filas={len(df)})")
    return out_path


# -------------------------------- (3) split_data -------------------------------- #
def split_data(
    folder_name: str | None = None,
    target_col: str = 'HiringDecision',
    test_size: float = 0.20,
    random_state: int = 42,
    **kwargs
) -> tuple[str, str]:
    run_folder = _resolve_folder_name(folder_name, **kwargs)
    merged_path = os.path.join(run_folder, 'preprocessed', 'data_merged.csv')
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"[split_data] No existe {merged_path}. Ejecute load_ands_merge primero.")

    df = pd.read_csv(merged_path)
    if target_col not in df.columns:
        raise ValueError(f"[split_data] No se encuentra la columna objetivo '{target_col}' en {merged_path}.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify_arg = y if 1 < y.nunique() < len(y) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    splits_dir = os.path.join(run_folder, 'splits')
    _ensure_dir(splits_dir)

    train_path = os.path.join(splits_dir, 'train.csv')
    test_path = os.path.join(splits_dir, 'test.csv')

    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)

    print(f"[split_data] Guardados: {train_path} (n={len(X_train)}), {test_path} (n={len(X_test)})")
    return train_path, test_path


# -------------------------------- (4) train_model -------------------------------- #
def train_model(
    model,
    folder_name: str | None = None,
    target_col: str = 'HiringDecision',
    **kwargs
) -> str:
    run_folder = _resolve_folder_name(folder_name, **kwargs)
    train_path = os.path.join(run_folder, 'splits', 'train.csv')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"[train_model] No existe {train_path}. Ejecute split_data primero.")

    df_train = pd.read_csv(train_path)
    if target_col not in df_train.columns:
        raise ValueError(f"[train_model] No se encuentra la columna objetivo '{target_col}' en {train_path}.")

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    num_cols, cat_cols = _infer_columns(df_train, target_col)

    num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    try:
        cat_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    except TypeError:
        cat_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        remainder='drop'
    )

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', model)
    ])

    pipe.fit(X_train, y_train)

    models_dir = os.path.join(run_folder, 'models')
    _ensure_dir(models_dir)

    model_name = model.__class__.__name__
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(models_dir, f"{model_name}_{stamp}.joblib")

    joblib.dump(pipe, out_path)
    print(f"[train_model] Modelo guardado en: {out_path}")
    return out_path


# -------------------------------- (5) evaluate_models -------------------------------- #
def evaluate_models(
    folder_name: str | None = None,
    target_col: str = 'HiringDecision',
    **kwargs
) -> tuple[str, float, str]:
    run_folder = _resolve_folder_name(folder_name, **kwargs)
    test_path = os.path.join(run_folder, 'splits', 'test.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"[evaluate_models] No existe {test_path}. Ejecute split_data y entrene modelos primero.")

    df_test = pd.read_csv(test_path)
    if target_col not in df_test.columns:
        raise ValueError(f"[evaluate_models] No se encuentra la columna objetivo '{target_col}' en {test_path}.")

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    models_dir = os.path.join(run_folder, 'models')
    candidates = glob.glob(os.path.join(models_dir, "*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"[evaluate_models] No se encontraron modelos .joblib en {models_dir}.")

    best_name, best_acc, best_path = None, -1.0, None

    for mpath in candidates:
        try:
            pipe = joblib.load(mpath)
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"[evaluate_models] {os.path.basename(mpath)} -> accuracy={acc:.4f}")
            if acc > best_acc:
                best_name, best_acc, best_path = os.path.basename(mpath), acc, mpath
        except Exception as e:
            print(f"[evaluate_models] Error evaluando {mpath}: {e}")

    if best_path is None:
        raise RuntimeError("[evaluate_models] No fue posible evaluar ningún modelo válido.")

    best_copy = os.path.join(models_dir, "best_model.joblib")
    shutil.copy(best_path, best_copy)

    print(f"[evaluate_models] MEJOR MODELO: {best_name} | accuracy={best_acc:.4f}")
    print(f"[evaluate_models] Copia guardada en: {best_copy}")
    return best_name, best_acc, best_copy


# -------------------------- (uso local / ejemplo) -------------------------- #
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    # Indica dónde están tus data_1.csv y data_2.csv originales:
    run_folder = create_folders(
        src_path1="data_1.csv",                  # <-- ajusta ruta
        src_path2="data_2.csv"                   # <-- opcional, ajusta ruta si existe
    )

    load_ands_merge(run_folder)
    split_data(run_folder)

    train_model(RandomForestClassifier(n_estimators=300, random_state=42), run_folder)
    train_model(LogisticRegression(max_iter=500), run_folder)

    evaluate_models(run_folder)
