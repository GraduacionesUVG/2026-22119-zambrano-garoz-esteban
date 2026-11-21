"""
Entrenamiento del modelo GRU baseline. La evaluación y el barrido de umbrales
viven en src/models/gru/evaluation.py.

Funciones puras y reutilizables desde notebook o script — ningún import de
este módulo carga datos, entrena, ni escribe archivos por sí solo. El punto
de entrada real es el notebook (notebooks/gru/01_baseline.ipynb), que
orquesta el pipeline completo: cargar .npy -> compute_class_weights ->
fijar_semillas -> build_gru_model -> train_gru_model -> plot_training_curves.

Uso típico:
    from src.models.gru.model import build_gru_model
    from src.models.gru.train import (
        compute_class_weights, fijar_semillas, train_gru_model,
        plot_training_curves,
    )

    class_weight_dict = compute_class_weights(y_train)
    fijar_semillas(0)
    model = build_gru_model(input_shape=(10, 59))
    history = train_gru_model(model, X_train, y_train, X_val, y_val, class_weight_dict)
    plot_training_curves(history, save_path="results/gru/01_baseline/baseline_w10_seed0_curves.png")
"""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.models import Model


def fijar_semillas(seed: int) -> None:
    """
    Fija todas las fuentes de aleatoriedad del entrenamiento: random de Python,
    NumPy y TensorFlow (keras.utils.set_random_seed), y activa las operaciones
    deterministas de TensorFlow.

    Debe llamarse antes de build_gru_model(): en Keras 3 los inicializadores de
    pesos y el Dropout toman su semilla de random de Python al construirse el
    modelo, así que np.random.seed + tf.random.set_seed no bastan.
    """
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """
    Calcula los pesos de clase balanceados para compensar el desbalance
    benigno/ataque (ADR 0013).

    Usa sklearn.utils.class_weight.compute_class_weight(class_weight='balanced'),
    que pondera cada clase inversamente proporcional a su frecuencia.

    Parameters
    ----------
    y_train : np.ndarray
        Etiquetas binarias de entrenamiento (0/1).

    Returns
    -------
    dict[int, float]
        Mapeo clase -> peso, listo para pasar como class_weight en model.fit().
    """
    clases = np.unique(y_train)
    pesos = compute_class_weight(class_weight="balanced", classes=clases, y=y_train)
    return {int(clase): float(peso) for clase, peso in zip(clases, pesos)}


def train_gru_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weight_dict: dict[int, float],
    epochs: int = 50,
    batch_size: int = 64,
    patience: int = 10,
) -> History:
    """
    Entrena el modelo GRU con early stopping sobre val_loss.

    No fija semillas: para que la inicialización de pesos, el dropout y el
    orden de los batches sean reproducibles, llamar a fijar_semillas() antes
    de build_gru_model().

    Usa EarlyStopping(monitor='val_loss', mode='min', patience=patience,
    restore_best_weights=True) como criterio de parada: el recall a umbral
    fijo sobre los pocos ataques de validación oscila de época en época,
    mientras que val_loss se calcula sobre probabilidades continuas y todas
    las filas de validación (ADR 0013).

    Parameters
    ----------
    model : tf.keras.Model
        Modelo compilado, típicamente build_gru_model(input_shape), construido
        después de fijar_semillas().
    X_train, y_train : np.ndarray
        Datos de entrenamiento.
    X_val, y_val : np.ndarray
        Datos de validación (para early stopping, no se usan para ajustar pesos).
    class_weight_dict : dict[int, float]
        Salida de compute_class_weights(y_train).
    epochs : int
        Épocas máximas (default 50; early stopping normalmente corta antes).
    batch_size : int
        Tamaño de batch (default 64).
    patience : int
        Épocas sin mejora de val_loss antes de detener el entrenamiento
        (default 10; ver ADR 0013).

    Returns
    -------
    tf.keras.callbacks.History
        Historial de entrenamiento (loss/accuracy/precision/recall/auc por
        época, train y val) — insumo de plot_training_curves().
    """
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Con patience >= 1, stopped_epoch > 0 identifica sin ambigüedad una parada
    # temprana real: la primera parada posible ocurre tras `patience` épocas sin
    # mejora después de la primera, es decir, en la época (0-indexada) `patience`.
    if early_stopping.stopped_epoch > 0:
        # stopped_epoch está 0-indexada; se suma 1 para reportar en el mismo formato
        # que el log "Epoch X/Y". La mejor época es la detenida menos `patience`.
        epoca_detenida = early_stopping.stopped_epoch + 1
        epoca_restaurada = epoca_detenida - early_stopping.patience
        mejor_val_loss = float(early_stopping.best)
        print(
            f"EarlyStopping: entrenamiento detenido en época {epoca_detenida}, "
            f"pesos restaurados de la época {epoca_restaurada} "
            f"(mejor val_loss: {mejor_val_loss:.4f})"
        )
    else:
        print(
            f"EarlyStopping no se activó: se completaron las {epochs} épocas "
            "solicitadas sin que val_loss dejara de mejorar el tiempo suficiente."
        )

    return history


def plot_training_curves(history: History, save_path: str | None = None) -> Figure:
    """
    Grafica loss y recall (train vs val) por época a partir del history de
    train_gru_model().

    Parameters
    ----------
    history : tf.keras.callbacks.History
        Salida de train_gru_model().
    save_path : str, opcional
        Si se pasa, guarda la figura como PNG en esa ruta (crea las carpetas
        intermedias si no existen). Si no, solo devuelve la figura para
        mostrarla inline en notebook.

    Returns
    -------
    matplotlib.figure.Figure
    """
    hist = history.history
    epocas = range(1, len(hist["loss"]) + 1)

    fig, (ax_loss, ax_recall) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epocas, hist["loss"], label="train")
    ax_loss.plot(epocas, hist["val_loss"], label="val")
    ax_loss.set_title("Loss por época")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    ax_recall.plot(epocas, hist["recall"], label="train")
    ax_recall.plot(epocas, hist["val_recall"], label="val")
    ax_recall.set_title("Recall por época")
    ax_recall.set_xlabel("Época")
    ax_recall.set_ylabel("Recall")
    ax_recall.legend()

    fig.tight_layout()

    if save_path:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig
