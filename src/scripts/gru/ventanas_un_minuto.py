"""
Registra en results/gru/00_datos/ventanas_un_minuto.json qué proporción de las
ventanas W=10 está formada solo por flujos del mismo minuto (Timestamp) que su fila
objetivo, por partición y en las filas comunes de test (idx_test_comun.npy).

Recorre las particiones como build_sequences(): ventanas de W filas consecutivas que
no cruzan bloque ni partición. El resultado es el mismo en las variantes csv y
aleatorio, porque la permutación ocurre dentro de cada minuto y no cambia el minuto
de cada posición (ADR 0012).

Uso:
    python scripts/gru/ventanas_un_minuto.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # no dejar __pycache__ en el repo al importar src

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.clean import (  # noqa: E402
    cargar_dataset_limpio,
    train_test_split_by_flow,
)

W = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--processed-path", default="data/processed/wednesday_low_and_slow_clean.csv",
    )
    parser.add_argument("--sequences-dir", default="data/sequences")
    parser.add_argument(
        "--idx-comun", default="results/gru/03_evaluacion_test/idx_test_comun.npy",
    )
    parser.add_argument("--output-dir", default="results/gru/00_datos")
    return parser.parse_args()


def un_solo_minuto(X: pd.DataFrame, window_size: int) -> pd.Series:
    """Por fila objetivo (índice de X), True si su ventana cae en un único minuto."""
    resultado = {}
    for _, bloque in X.groupby("bloque", sort=False):
        ts = bloque["Timestamp"].to_numpy()
        for k in range(window_size - 1, len(bloque)):
            resultado[bloque.index[k]] = bool(ts[k - window_size + 1] == ts[k])
    return pd.Series(resultado)


def main() -> None:
    args = parse_args()
    df = cargar_dataset_limpio(args.processed_path)
    X_train, X_val, X_test, *_ = train_test_split_by_flow(df)

    salida = {"window_size": W, "por_particion": {}}
    for nombre, X in (("train", X_train), ("val", X_val), ("test", X_test)):
        mismo = un_solo_minuto(X, W)
        # Mismas filas objetivo que las secuencias guardadas, en el mismo orden
        idx = np.load(Path(args.sequences_dir) / f"idx_{nombre}_w{W}.npy")
        if not np.array_equal(mismo.index.to_numpy(), idx):
            raise RuntimeError(f"Las ventanas de {nombre} no coinciden con idx_{nombre}_w{W}.npy")
        salida["por_particion"][nombre] = {
            "ventanas": int(len(mismo)),
            "en_un_solo_minuto": int(mismo.sum()),
            "porcentaje": round(100 * float(mismo.mean()), 2),
        }
        if nombre == "test":
            comun = mismo.loc[np.load(args.idx_comun)]
            salida["test_filas_comunes"] = {
                "ventanas": int(len(comun)),
                "en_un_solo_minuto": int(comun.sum()),
                "porcentaje": round(100 * float(comun.mean()), 2),
            }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ventanas_un_minuto.json"
    path.write_text(json.dumps(salida, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{path}:\n{json.dumps(salida, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
