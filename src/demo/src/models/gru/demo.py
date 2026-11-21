"""
Inferencia del GRU para la demo: reproduce el pipeline de la tesis sobre flujos
sin escalar.

    flujos crudos -> sign(x)*log1p(|x|) -> StandardScaler (ajustado en train)
    -> ventanas de W flujos por bloque (paso 1) -> GRU -> probabilidad de ataque
    -> umbral F2 elegido en validación

Los artefactos los genera scripts/gru/exportar_demo.py de este repositorio (se ejecuta desde src/).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def cargar_config(model_dir: str | Path) -> dict:
    ruta = Path(model_dir) / "gru_config.json"
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta scripts/gru/exportar_demo.py desde src/."
        )
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def cargar_modelo(ruta: str | Path):
    import keras

    return keras.models.load_model(str(ruta), compile=False)


def cargar_scaler(model_dir: str | Path):
    return joblib.load(Path(model_dir) / "scaler_csv.joblib")


def escalar_flujos(x: pd.DataFrame, scaler, caracteristicas: list[str]) -> np.ndarray:
    faltan = [c for c in caracteristicas if c not in x.columns]
    if faltan:
        raise ValueError(f"Faltan características en X_test: {faltan[:10]}")

    datos = x[caracteristicas].astype(float)
    if not np.isfinite(datos.to_numpy()).all():
        raise ValueError("X_test contiene NaN o infinitos.")

    log = np.sign(datos) * np.log1p(np.abs(datos))
    return scaler.transform(log).astype(np.float32)


def construir_ventanas(
    escalado: np.ndarray, bloques: np.ndarray, ventana: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ventanas deslizantes de `ventana` flujos que no cruzan bloques. Devuelve las
    ventanas (n, W, F) y la posición (fila de X_test) del flujo objetivo de cada una.
    """
    ventanas, objetivos = [], []
    for bloque in pd.unique(bloques):
        pos = np.flatnonzero(bloques == bloque)
        if not np.array_equal(pos, np.arange(pos[0], pos[-1] + 1)):
            raise ValueError(f"Las filas del bloque {bloque} no son contiguas en X_test.")
        if len(pos) < ventana:
            continue
        v = np.lib.stride_tricks.sliding_window_view(escalado[pos], ventana, axis=0)
        ventanas.append(v.transpose(0, 2, 1))
        objetivos.append(pos[ventana - 1:])

    if not ventanas:
        raise ValueError(f"Ningún bloque tiene al menos {ventana} flujos.")
    return np.ascontiguousarray(np.concatenate(ventanas)), np.concatenate(objetivos)


def predecir(model, ventanas: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    return np.asarray(model.predict(ventanas, batch_size=batch_size, verbose=0)).reshape(-1)