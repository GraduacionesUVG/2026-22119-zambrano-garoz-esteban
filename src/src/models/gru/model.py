"""
Arquitectura del modelo GRU baseline.

Solo define y compila el modelo — no carga datos ni entrena nada al importar
este módulo. El entrenamiento/evaluación vive en src/models/gru/train.py.

Uso típico:
    from src.models.gru.model import build_gru_model

    model = build_gru_model(input_shape=(10, 59))
    model.summary()
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_gru_model(
    input_shape: tuple[int, int],
    units: int = 32,
    second_layer_units: int | None = None,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
) -> Model:
    """
    Construye y compila un modelo GRU. Defaults reproducen el baseline
    exacto: Input -> GRU(32) -> Dropout(0.3) -> Dense(1, sigmoid).

    Con second_layer_units: Input -> GRU(units, return_sequences=True) ->
    GRU(second_layer_units) -> Dropout(dropout) -> Dense(1, sigmoid).

    Clasificación binaria (0 = benigno, 1 = ataque Slowloris/Slowhttptest).

    Parameters
    ----------
    input_shape : tuple[int, int]
        (window_size, n_features) de las secuencias generadas por
        build_sequences() en src/models/gru/sequences.py. Para W=10 con las
        59 columnas de FEATURE_COLUMNS, es (10, 59).
    units : int
        Unidades de la (primera) capa GRU.
    second_layer_units : int, opcional
        Si se especifica, agrega una segunda capa GRU con esta cantidad de
        unidades. Default None (arquitectura de una sola capa).
    dropout : float
        Tasa de Dropout después de la(s) capa(s) GRU.
    learning_rate : float
        Tasa de aprendizaje de Adam.

    Returns
    -------
    tf.keras.Model
        Modelo compilado con Adam(learning_rate), binary_crossentropy y métricas de
        accuracy/precision/recall/AUC — listo para .fit(), sin entrenar.
    """
    inputs = Input(shape=input_shape)
    if second_layer_units is None:
        x = GRU(units=units, return_sequences=False)(inputs)
    else:
        x = GRU(units=units, return_sequences=True)(inputs)
        x = GRU(units=second_layer_units, return_sequences=False)(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
