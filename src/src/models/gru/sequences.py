"""
Construcción de secuencias temporales para el GRU a partir de X_train/X_val/X_test
(salida de train_test_split_by_flow() + scale_features() en src/preprocessing/clean.py).

Específico del modelo GRU — no lo usan SVM/XGBoost, por eso vive aquí y no en
src/preprocessing/clean.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Features de cada paso de la secuencia: todas las columnas del CSV procesado
# (data/processed/wednesday_low_and_slow_clean.csv) excepto Timestamp, Label y
# label_binaria, en el mismo orden. 'bloque' tampoco entra (metadata, ver
# build_sequences).
FEATURE_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]
assert len(FEATURE_COLUMNS) == 59


def build_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    window_size: int,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convierte un DataFrame de un solo split (train, val o test) en secuencias
    deslizantes (sliding window, stride 1) para el GRU.

    Para cada fila objetivo t, la secuencia son las window_size filas que
    terminan en t (incluida t); la etiqueta es y en esa misma posición. Las
    primeras window_size - 1 filas de cada bloque se descartan por falta de
    historia.

    Se llama por separado sobre cada split (build_sequences(X_train, y_train, W),
    build_sequences(X_val, y_val, W), build_sequences(X_test, y_test, W)). Una
    ventana solo se construye con filas del mismo 'bloque' — como X ya viene
    filtrado a un único split, agrupar solo por 'bloque' basta para nunca cruzar
    ni split ni bloque (la columna 'split' es interna a train_test_split_by_flow
    y nunca llega a X).

    Parameters
    ----------
    X : pd.DataFrame
        X_train, X_val o X_test (salida de train_test_split_by_flow() /
        scale_features()). Debe incluir 'Timestamp' y 'bloque' como metadata.
    y : pd.Series
        y_train, y_val o y_test correspondiente, mismo índice que X.
    window_size : int
        Cantidad de pasos temporales por secuencia (W).
    feature_columns : list[str], opcional
        Columnas numéricas a usar como features. Default: FEATURE_COLUMNS
        (todo lo de X excepto Timestamp/bloque).

    Returns
    -------
    X_seq : np.ndarray, dtype float32, shape (n_secuencias, window_size, n_features)
    y_seq : np.ndarray, dtype int8, shape (n_secuencias,)
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    for col in ("Timestamp", "bloque"):
        if col not in X.columns:
            raise ValueError(
                f"'{col}' no está en X — se necesita para ordenar/agrupar las "
                "secuencias. ¿Viene X de train_test_split_by_flow()?"
            )

    orden_original = pd.Series(np.arange(len(X)), index=X.index)
    orden_esperado = X["Timestamp"].argsort(kind="stable")
    assert np.array_equal(orden_original.to_numpy(), orden_esperado.to_numpy()), (
        "X no está ordenado cronológicamente por Timestamp (con el índice "
        "original como desempate) — build_sequences() asume ese orden para "
        "poder ventanear por posición. Revisa que X venga tal cual de "
        "train_test_split_by_flow(), sin reordenar."
    )

    secuencias = []
    etiquetas = []

    for bloque in sorted(X["bloque"].unique()):
        mask = X["bloque"] == bloque
        # float32: precisión suficiente para features ya escaladas (media 0,
        # std 1) y la mitad de memoria que el float64 default de pandas —
        # relevante porque cada fila se repite en hasta window_size secuencias
        X_bloque = X.loc[mask, feature_columns].to_numpy(dtype=np.float32)
        y_bloque = y.loc[mask].to_numpy()
        n = len(X_bloque)

        n_secuencias_bloque = n - window_size + 1
        if n_secuencias_bloque <= 0:
            print(
                f"ADVERTENCIA: bloque {bloque} tiene {n} filas, insuficientes "
                f"para window_size={window_size} (se necesitan al menos "
                f"{window_size}) — 0 secuencias generadas para este bloque."
            )
            continue

        for fin in range(window_size - 1, n):
            inicio = fin - window_size + 1
            secuencias.append(X_bloque[inicio:fin + 1])
            etiquetas.append(y_bloque[fin])

    if not secuencias:
        return (
            np.empty((0, window_size, len(feature_columns)), dtype=np.float32),
            np.empty((0,), dtype=np.int8),
        )

    X_seq = np.stack(secuencias)
    # int8 basta: label_binaria es 0/1 (ver encode_labels() en clean.py), y_train
    # llega como int64 pero cargar ese ancho para un binario es puro desperdicio
    # de memoria a esta escala (cientos de miles de secuencias)
    y_seq = np.array(etiquetas, dtype=np.int8)
    return X_seq, y_seq
