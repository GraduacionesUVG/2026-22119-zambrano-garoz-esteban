"""
Exporta el modelo principal del GRU (W=10, orden aleatorio) y sus datos de prueba
a la demo interactiva (src/demo/).

Solo se exporta una semilla (SEMILLA): la más cercana a la media de test de
las cinco en recall, precision y F1. No es la de mejores métricas, para que la
demo muestre un resultado representativo de lo que reporta la tesis.

Escribe en la carpeta de la demo:
    Models/GRU/aleatorio_w10_seed{SEMILLA}.keras
    Models/GRU/scaler_csv.joblib
    Models/GRU/gru_config.json        ventana, features, umbral de validación,
                                      métricas de test esperadas y media de la tesis
    Data_Test/gru_test/X_test.csv     flujos de test SIN escalar, en el orden
                                      aleatorio de las secuencias, con 'bloque'
    Data_Test/gru_test/y_test.csv     label_binaria
    Data_Test/gru_test/meta_test.csv  idx_fila, Timestamp, Label, en_comun

El orden de las filas se reconstruye desde X_test_w10_aleatorio.npy e
idx_test_w10_aleatorio.npy, y se verifica que, al aplicar log con signo +
scaler_csv.joblib y ventanear por bloque, se obtienen exactamente esas secuencias.

Uso (desde src/):
    python scripts/gru/exportar_demo.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.models.gru.sequences import FEATURE_COLUMNS  # noqa: E402
from src.preprocessing.clean import (  # noqa: E402
    _log_con_signo,
    cargar_dataset_limpio,
    train_test_split_by_flow,
)

MODELO = "aleatorio_w10"
SEMILLA = 3

PROCESSED_PATH = RAIZ / "data/processed/wednesday_low_and_slow_clean.csv"
SEQ_DIR = RAIZ / "data/sequences"
MODELS_DIR = RAIZ / "models/gru/02_ablaciones"
SEEDS_PATH = RAIZ / "results/gru/02_ablaciones/aleatorio_w10_seeds.json"
TEST_DIR = RAIZ / "results/gru/03_evaluacion_test"

METRICAS_CONFIG = ["tp", "fp", "fn", "tn", "recall", "precision", "f1", "roc_auc", "pr_auc"]


def reconstruir_orden(
    X_test: pd.DataFrame, escalado: pd.DataFrame, X_seq: np.ndarray, idx_seq: np.ndarray,
) -> list[int]:
    """
    Orden de las filas de test que generó las secuencias: por bloque, las W-1 filas
    iniciales (sin secuencia propia) seguidas de las filas objetivo en el orden de
    idx_seq. Las filas iniciales se identifican por sus features escaladas en la
    primera ventana del bloque.
    """
    W = X_seq.shape[1]
    bloque_objetivo = X_test.loc[idx_seq, "bloque"].to_numpy()

    orden: list[int] = []
    for bloque in pd.unique(bloque_objetivo):
        pos = np.flatnonzero(bloque_objetivo == bloque)
        assert np.array_equal(pos, np.arange(pos[0], pos[-1] + 1)), (
            f"Las secuencias del bloque {bloque} no son contiguas"
        )
        objetivos = idx_seq[pos]
        filas_bloque = X_test.index[X_test["bloque"] == bloque]
        candidatas = list(filas_bloque.difference(objetivos))
        assert len(candidatas) == W - 1, (
            f"Bloque {bloque}: se esperaban {W - 1} filas sin secuencia y hay {len(candidatas)}"
        )

        primera = X_seq[pos[0]]
        assert np.allclose(escalado.loc[objetivos[0]].to_numpy(), primera[-1], atol=1e-5)

        iniciales = []
        for paso in range(W - 1):
            coincide = [
                c for c in candidatas
                if np.allclose(escalado.loc[c].to_numpy(), primera[paso], atol=1e-5)
            ]
            if not coincide:
                raise RuntimeError(f"Bloque {bloque}: ninguna fila coincide con el paso {paso}")
            iniciales.append(coincide[0])
            candidatas.remove(coincide[0])

        orden.extend(iniciales + [int(i) for i in objetivos])
    return orden


def verificar_secuencias(
    escalado_ordenado: np.ndarray, bloques: np.ndarray, X_seq: np.ndarray,
) -> None:
    W = X_seq.shape[1]
    inicio = 0
    for bloque in pd.unique(bloques):
        filas = escalado_ordenado[bloques == bloque].astype(np.float32)
        ventanas = np.lib.stride_tricks.sliding_window_view(filas, W, axis=0).transpose(0, 2, 1)
        fin = inicio + len(ventanas)
        assert np.allclose(ventanas, X_seq[inicio:fin], atol=1e-5), (
            f"Las ventanas reconstruidas del bloque {bloque} no coinciden con el .npy"
        )
        inicio = fin
    assert inicio == len(X_seq), f"Se reconstruyeron {inicio} ventanas de {len(X_seq)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--destino", default="demo", help="Carpeta de la demo (por defecto, demo/)")
    args = parser.parse_args()

    destino = Path(args.destino).resolve()
    if not (destino / "app.py").is_file():
        raise SystemExit(f"No se encontró app.py en {destino}")
    dir_modelos = destino / "Models" / "GRU"
    dir_datos = destino / "Data_Test" / "gru_test"
    dir_modelos.mkdir(parents=True, exist_ok=True)
    dir_datos.mkdir(parents=True, exist_ok=True)

    # 1. Partición de test con el índice de idx_*.npy (desempate csv)
    df = cargar_dataset_limpio(str(PROCESSED_PATH))
    _, _, X_test, _, _, y_test, _, _, meta_test = train_test_split_by_flow(df, return_meta=True)
    del df

    # 2. Secuencias guardadas de la variante aleatoria
    X_seq = np.load(SEQ_DIR / "X_test_w10_aleatorio.npy")
    y_seq = np.load(SEQ_DIR / "y_test_w10_aleatorio.npy")
    idx_seq = np.load(SEQ_DIR / "idx_test_w10_aleatorio.npy")
    idx_comun = np.load(TEST_DIR / "idx_test_comun.npy")
    scaler = joblib.load(SEQ_DIR / "scaler_csv.joblib")
    assert list(scaler.feature_names_in_) == FEATURE_COLUMNS
    assert np.array_equal(y_test.loc[idx_seq].to_numpy(), y_seq)

    escalado = pd.DataFrame(
        scaler.transform(_log_con_signo(X_test[FEATURE_COLUMNS].astype(float))),
        index=X_test.index, columns=FEATURE_COLUMNS,
    )

    # 3. Orden de las filas y verificación contra las secuencias
    orden = reconstruir_orden(X_test, escalado, X_seq, idx_seq)
    bloques = X_test.loc[orden, "bloque"].to_numpy()
    verificar_secuencias(escalado.loc[orden].to_numpy(), bloques, X_seq)
    print(f"OK: {len(orden)} filas reproducen las {len(X_seq)} secuencias de test (aleatorio).")

    # 4. Datos de test sin escalar, en ese orden
    X_out = X_test.loc[orden, ["bloque"] + FEATURE_COLUMNS].reset_index(drop=True)
    y_out = y_test.loc[orden].rename("label_binaria").reset_index(drop=True)
    meta_out = pd.DataFrame({
        "idx_fila": orden,
        "Timestamp": X_test.loc[orden, "Timestamp"].to_numpy(),
        "Label": meta_test.loc[orden, "Label"].to_numpy(),
        "en_comun": np.isin(orden, idx_comun),
    })
    assert int(meta_out["en_comun"].sum()) == len(idx_comun)

    X_out.to_csv(dir_datos / "X_test.csv", index=False)
    y_out.to_csv(dir_datos / "y_test.csv", index=False)
    meta_out.to_csv(dir_datos / "meta_test.csv", index=False)

    releido = pd.read_csv(dir_datos / "X_test.csv")
    assert np.array_equal(releido[FEATURE_COLUMNS].to_numpy(float), X_out[FEATURE_COLUMNS].to_numpy(float)), (
        "El CSV escrito no conserva los valores exactos"
    )

    # 5. Modelos, escalador y configuración
    with open(SEEDS_PATH, encoding="utf-8") as f:
        umbral = {fila["semilla"]: fila["umbral_f2"] for fila in json.load(f)["por_semilla"]}[SEMILLA]
    with open(TEST_DIR / "test_resultados.json", encoding="utf-8") as f:
        test = json.load(f)
    filas_modelo = [fila for fila in test["por_semilla"] if fila["modelo"] == MODELO]
    fila_test = next(fila for fila in filas_modelo if fila["semilla"] == SEMILLA)
    assert fila_test["umbral_f2_val"] == umbral

    archivo = f"{MODELO}_seed{SEMILLA}.keras"
    shutil.copy2(MODELS_DIR / archivo, dir_modelos / archivo)
    shutil.copy2(SEQ_DIR / "scaler_csv.joblib", dir_modelos / "scaler_csv.joblib")

    config = {
        "modelo": MODELO,
        "nombre": "GRU W=10 (orden aleatorio)",
        "ventana": int(X_seq.shape[1]),
        "caracteristicas": FEATURE_COLUMNS,
        "preprocesamiento": "sign(x)*log1p(|x|) + StandardScaler (scaler_csv.joblib)",
        "semilla": SEMILLA,
        "archivo": archivo,
        "umbral": umbral,
        "criterio_umbral": "F2 máximo en validación",
        "n_filas_comunes": int(len(idx_comun)),
        "test_esperado": {k: fila_test[k] for k in METRICAS_CONFIG},
        "n_semillas_tesis": len(filas_modelo),
        "test_media": {k: test["media"][MODELO][k] for k in METRICAS_CONFIG},
        "test_std": {k: test["std"][MODELO][k] for k in METRICAS_CONFIG},
    }
    with open(dir_modelos / "gru_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Modelos y configuración -> {dir_modelos}")
    print(f"Datos de test            -> {dir_datos}")
    print(
        f"Filas: {len(X_out)} | ataques: {int(y_out.sum())} | "
        f"filas comunes: {int(meta_out['en_comun'].sum())}"
    )


if __name__ == "__main__":
    main()