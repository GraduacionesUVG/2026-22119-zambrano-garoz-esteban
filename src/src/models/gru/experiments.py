"""
Ejecución de un experimento del GRU con varias semillas: entrena una
configuración por semilla, guarda sus artefactos y resume las métricas de
validación (fila por semilla, media y desviación estándar).

Uso típico (ver notebooks/gru/01_baseline.ipynb):
    from src.models.gru.experiments import correr_experimento

    config = {"units": 32, "dropout": 0.3, "learning_rate": 1e-3,
              "batch_size": 64, "epochs": 50, "patience": 10}
    tabla = correr_experimento(
        "baseline_w10", config, X_train, y_train, X_val, y_val,
        semillas=[0, 1, 2, 3, 4],
        models_dir="models/gru/01_baseline", results_dir="results/gru/01_baseline",
    )
"""

from __future__ import annotations

import json
import os
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.gru.evaluation import metricas_desde_probabilidades
from src.models.gru.model import build_gru_model
from src.models.gru.train import (
    compute_class_weights,
    fijar_semillas,
    plot_training_curves,
    train_gru_model,
)


def correr_experimento(
    nombre: str,
    config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    semillas: list[int],
    models_dir: str,
    results_dir: str,
) -> pd.DataFrame:
    """
    Entrena la configuración `config` una vez por semilla y evalúa cada modelo
    en validación (nunca en test).

    Por semilla guarda {models_dir}/{nombre}_seed{seed}.keras,
    {results_dir}/{nombre}_seed{seed}_curves.png y
    {results_dir}/{nombre}_seed{seed}_val_proba.npy. Al final guarda
    {results_dir}/{nombre}_seeds.json con config, semillas, por_semilla, media y std.

    Parameters
    ----------
    nombre : str
        Prefijo de los archivos generados (ej. "baseline_w10").
    config : dict
        Claves: units, dropout, learning_rate, batch_size, epochs, patience y,
        opcionalmente, second_layer_units.
    X_train, y_train, X_val, y_val : np.ndarray
        Secuencias y etiquetas de entrenamiento y validación.
    semillas : list[int]
        Una corrida por semilla (fijar_semillas() antes de construir cada modelo).
    models_dir, results_dir : str
        Carpetas de salida; se crean si no existen.

    Returns
    -------
    pd.DataFrame
        Indexado por semilla, más las filas "media" y "std" (ddof=1). Columnas:
        val_loss, mejor_epoca, epocas_entrenadas, tiempo_entrenamiento_s y las de
        metricas_desde_probabilidades().
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    class_weight_dict = compute_class_weights(y_train)

    filas = []
    for seed in semillas:
        print(f"\n{'=' * 70}\n{nombre} — semilla {seed}\n{'=' * 70}")
        keras.backend.clear_session()
        fijar_semillas(seed)
        model = build_gru_model(
            X_train.shape[1:],
            units=config["units"],
            second_layer_units=config.get("second_layer_units"),
            dropout=config["dropout"],
            learning_rate=config["learning_rate"],
        )

        inicio = time.perf_counter()
        history = train_gru_model(
            model, X_train, y_train, X_val, y_val, class_weight_dict,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            patience=config["patience"],
        )
        tiempo_entrenamiento_s = time.perf_counter() - inicio

        model.save(f"{models_dir}/{nombre}_seed{seed}.keras")
        fig = plot_training_curves(
            history, save_path=f"{results_dir}/{nombre}_seed{seed}_curves.png",
        )
        plt.close(fig)

        val_loss = float(model.evaluate(X_val, y_val, verbose=0)[0])
        proba = model.predict(X_val, verbose=0).ravel()
        np.save(f"{results_dir}/{nombre}_seed{seed}_val_proba.npy", proba)

        val_loss_hist = history.history["val_loss"]
        filas.append({
            "semilla": seed,
            "val_loss": val_loss,
            "mejor_epoca": int(np.argmin(val_loss_hist)) + 1,
            "epocas_entrenadas": len(val_loss_hist),
            "tiempo_entrenamiento_s": tiempo_entrenamiento_s,
            **metricas_desde_probabilidades(y_val, proba),
        })
        print(filas[-1])

        del model, history

    tabla = pd.DataFrame(filas).set_index("semilla")
    media = tabla.mean()
    std = tabla.std(ddof=1)

    resultado = {
        "modelo": nombre,
        "conjunto": "validacion",
        "config": config,
        "semillas": list(semillas),
        "por_semilla": filas,
        "media": {k: float(v) for k, v in media.items()},
        "std": {k: float(v) for k, v in std.items()},
    }
    seeds_path = f"{results_dir}/{nombre}_seeds.json"
    with open(seeds_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en: {seeds_path}")

    return pd.concat([tabla, media.to_frame("media").T, std.to_frame("std").T])
