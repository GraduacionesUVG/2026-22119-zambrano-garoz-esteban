"""
Registra en results/gru/00_datos/ dos cifras derivadas que no quedan en ningún output:

- conteos_limpieza.json: filas tras cada paso de load_and_clean(), llamando a las
  mismas funciones en el mismo orden. Comprueba que el resultado coincide con el CSV
  procesado.
- pesos_clase.json: pesos de clase de cada modelo GRU, con compute_class_weights()
  sobre el y_train que usa cada uno (W=1 usa el de la variante csv; ADR 0012).

Uso:
    python scripts/gru/conteos_limpieza_y_pesos.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # no dejar __pycache__ en el repo al importar src

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.gru.train import compute_class_weights  # noqa: E402
from src.preprocessing.clean import (  # noqa: E402
    basic_cleaning,
    cargar_dataset_limpio,
    encode_labels,
    excluir_ataques_fuera_de_ventana,
    excluir_respuestas_de_ataque,
    filter_low_and_slow,
    load_raw_csvs,
)

# Modelo -> archivo de y_train en data/sequences/
Y_TRAIN_POR_MODELO = {
    "baseline_w10": "y_train_w10.npy",
    "w1": "y_train_w10.npy",
    "aleatorio_w10": "y_train_w10_aleatorio.npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--processed-path", default="data/processed/wednesday_low_and_slow_clean.csv",
    )
    parser.add_argument("--sequences-dir", default="data/sequences")
    parser.add_argument("--output-dir", default="results/gru/00_datos")
    return parser.parse_args()


def conteos_limpieza(raw_dir: str, processed_path: str) -> dict:
    """Filas tras cada paso de load_and_clean() y eliminadas en cada uno."""
    pasos = [
        ("filtro_de_clases", filter_low_and_slow),
        ("exclusion_respuestas_de_ataque", excluir_respuestas_de_ataque),
        ("limpieza_basica", basic_cleaning),
        ("codificacion_de_etiquetas", encode_labels),
        ("exclusion_ataques_fuera_de_ventana", excluir_ataques_fuera_de_ventana),
    ]
    df = load_raw_csvs(raw_dir)
    filas = [{"paso": "captura_del_miercoles", "filas": len(df), "eliminadas": None}]
    for nombre, funcion in pasos:
        antes = len(df)
        df = funcion(df)
        filas.append({"paso": nombre, "filas": len(df), "eliminadas": antes - len(df)})

    procesado = cargar_dataset_limpio(processed_path)
    if len(procesado) != len(df) or not (
        procesado["Label"].value_counts().sort_index()
        .equals(df["Label"].value_counts().sort_index())
    ):
        raise RuntimeError(
            f"load_and_clean() da {len(df)} filas y el CSV procesado {len(procesado)}, "
            "o difieren los conteos por clase; regenerar el CSV antes de registrar."
        )
    return {
        "pasos": filas,
        "filas_por_clase": {k: int(v) for k, v in df["Label"].value_counts().items()},
    }


def pesos_clase(sequences_dir: str) -> dict:
    """Pesos de clase balanceados de cada modelo, calculados sobre su y_train."""
    pesos = {}
    for modelo, archivo in Y_TRAIN_POR_MODELO.items():
        y_train = np.load(Path(sequences_dir) / archivo)
        cw = compute_class_weights(y_train)
        pesos[modelo] = {
            "archivo_y_train": archivo,
            "secuencias": int(len(y_train)),
            "ataques": int(y_train.sum()),
            "peso_benigno": cw[0],
            "peso_ataque": cw[1],
        }
    return pesos


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    salidas = {
        "conteos_limpieza.json": conteos_limpieza(args.raw_dir, args.processed_path),
        "pesos_clase.json": pesos_clase(args.sequences_dir),
    }
    for nombre, contenido in salidas.items():
        path = output_dir / nombre
        path.write_text(json.dumps(contenido, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n{path}:\n{json.dumps(contenido, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
