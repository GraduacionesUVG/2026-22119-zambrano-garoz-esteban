"""
Estadísticas descriptivas por clase y categorías de las features del GRU.

Lee el CSV procesado y escribe en results/gru/00_datos/:
- estadisticas_por_clase.csv: n, p25, mediana y p75 de COLUMNAS_ESTADISTICAS por clase,
  en unidades originales (sin transformación ni escalado).
- categorias_features.csv: categoría de cada columna de FEATURE_COLUMNS.

Uso:
    python scripts/gru/estadisticas_descriptivas.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # no dejar __pycache__ en el repo al importar src

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.gru.sequences import FEATURE_COLUMNS  # noqa: E402
from src.preprocessing.clean import cargar_dataset_limpio  # noqa: E402

CATEGORIAS = {
    "duración de conexión": [
        "Flow Duration",
        "Active Mean", "Active Std", "Active Max", "Active Min",
    ],
    "número de paquetes": [
        "Total Fwd Packets", "Total Backward Packets", "act_data_pkt_fwd",
    ],
    "volumen de datos (bytes/longitudes)": [
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
        "Packet Length Variance", "Average Packet Size", "Fwd Header Length", "Bwd Header Length",
    ],
    "tasa de envío (por segundo)": [
        "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
    ],
    "intervalos entre paquetes (IAT)": [
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    ],
    "otras": [
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
        "URG Flag Count", "ECE Flag Count", "Down/Up Ratio",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward", "min_seg_size_forward",
    ],
}

COLUMNAS_ESTADISTICAS = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Fwd IAT Mean",
]
CLASES = ["BENIGN", "DoS slowloris", "DoS Slowhttptest"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estadísticas descriptivas por clase y categorías de las features del GRU.",
    )
    parser.add_argument(
        "--processed-path", default="data/processed/wednesday_low_and_slow_clean.csv",
        help="CSV procesado (salida del notebook de limpieza)",
    )
    parser.add_argument(
        "--results-dir", default="results/gru/00_datos",
        help="Carpeta donde guardar los CSV (default: results/gru/00_datos)",
    )
    return parser.parse_args()


def tabla_categorias() -> pd.DataFrame:
    """Una fila por columna de FEATURE_COLUMNS, en ese orden, con su categoría."""
    categoria_de = {col: cat for cat, cols in CATEGORIAS.items() for col in cols}
    asignadas = [col for cols in CATEGORIAS.values() for col in cols]
    repetidas = sorted({col for col in asignadas if asignadas.count(col) > 1})
    faltantes = sorted(set(FEATURE_COLUMNS) - set(asignadas))
    sobrantes = sorted(set(asignadas) - set(FEATURE_COLUMNS))
    if repetidas or faltantes or sobrantes:
        raise RuntimeError(
            f"CATEGORIAS no cubre FEATURE_COLUMNS una sola vez: repetidas {repetidas}, "
            f"faltantes {faltantes}, sobrantes {sobrantes}"
        )
    return pd.DataFrame({
        "columna": FEATURE_COLUMNS,
        "categoria": [categoria_de[col] for col in FEATURE_COLUMNS],
    })


def tabla_estadisticas(df: pd.DataFrame) -> pd.DataFrame:
    """n, p25, mediana y p75 de cada columna de COLUMNAS_ESTADISTICAS por clase."""
    faltantes = [c for c in COLUMNAS_ESTADISTICAS if c not in df.columns]
    if faltantes:
        raise RuntimeError(f"Faltan columnas en el CSV procesado: {faltantes}")

    filas = []
    for col in COLUMNAS_ESTADISTICAS:
        for clase in CLASES:
            valores = df.loc[df["Label"] == clase, col]
            filas.append({
                "columna": col,
                "clase": clase,
                "n": int(valores.notna().sum()),
                "p25": valores.quantile(0.25),
                "mediana": valores.median(),
                "p75": valores.quantile(0.75),
            })
    return pd.DataFrame(filas)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    categorias = tabla_categorias()
    df = cargar_dataset_limpio(args.processed_path)
    print(f"Filas: {len(df):,}")
    print(df["Label"].value_counts().reindex(CLASES).to_string())

    estadisticas = tabla_estadisticas(df)

    results_dir.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: para que Excel muestre bien las tildes de las categorías
    estadisticas_path = results_dir / "estadisticas_por_clase.csv"
    categorias_path = results_dir / "categorias_features.csv"
    estadisticas.to_csv(estadisticas_path, index=False, encoding="utf-8-sig")
    categorias.to_csv(categorias_path, index=False, encoding="utf-8-sig")
    print(f"\nGuardado: {estadisticas_path}")
    print(f"Guardado: {categorias_path}")

    print("\nColumnas por categoría:")
    conteo = categorias["categoria"].value_counts().reindex(CATEGORIAS)
    print(conteo.to_string())
    print(f"Total: {int(conteo.sum())}")


if __name__ == "__main__":
    main()
