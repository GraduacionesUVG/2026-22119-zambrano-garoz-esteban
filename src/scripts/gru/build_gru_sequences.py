"""
Script de reproducibilidad de punta a punta para las secuencias del GRU.

Encadena el pipeline compartido (cargar_dataset_limpio -> train_test_split_by_flow ->
scale_features -> build_sequences) y guarda en data/sequences/ los .npy de cada
variante (VARIANTES) y el scaler.

Hay un único split y un único scaler (desempate="csv"). La variante "aleatorio" es
una ablación: conserva exactamente las mismas filas en cada split y solo permuta
el orden de las filas dentro de cada (split, bloque, Timestamp), para medir cuánto
depende la GRU del orden del archivo dentro de cada minuto.

No reimplementa ninguna lógica de src/ -- solo la orquesta y verifica el resultado
antes de escribir nada: el split se valida contra los criterios de verificar_split()
(reporte en results/gru/split_verificacion.txt), la variante "aleatorio" se compara
fila a fila con "csv", y cada par X/y se valida (shape, NaN/Inf, conteo de ataques)
antes de np.save.

Uso:
    python scripts/gru/build_gru_sequences.py
    python scripts/gru/build_gru_sequences.py --processed-path data/processed/wednesday_low_and_slow_clean.csv
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# El repo no está empaquetado/instalado (igual que conftest.py en la raíz para
# pytest) -- al correr este archivo directo con `python scripts/gru/...py`,
# Python solo agrega su propia carpeta a sys.path, no la raíz del repo. Sin
# esto, `from src...` fallaría.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.gru.sequences import FEATURE_COLUMNS, build_sequences  # noqa: E402
from src.preprocessing.clean import (  # noqa: E402
    cargar_dataset_limpio,
    reportar_duplicados_de_features,
    scale_features,
    train_test_split_by_flow,
)

TRANSFORMACION_LOG = True  # igual que el pipeline del SVM (log con signo + StandardScaler)

# Orden dentro de cada minuto -> tamaños de ventana a generar. "csv" conserva el
# orden del archivo; "aleatorio" lo permuta (ver permutar_dentro_de_minuto())
VARIANTES = {"csv": (10,), "aleatorio": (10,)}
SEMILLA_PERMUTACION = 42

ATAQUES = ("DoS slowloris", "DoS Slowhttptest")
BLOQUES_ATAQUE = (2, 4)
BLOQUES_BENIGNOS = (1, 3, 5)
DURACION_MIN_VAL = pd.Timedelta(minutes=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera las secuencias del GRU a partir del CSV procesado, de punta a punta.",
    )
    parser.add_argument(
        "--processed-path", default="data/processed/wednesday_low_and_slow_clean.csv",
        help="CSV procesado (salida del notebook de limpieza)",
    )
    parser.add_argument(
        "--output-dir", default="data/sequences",
        help="Carpeta donde guardar los .npy y los scalers (default: data/sequences)",
    )
    parser.add_argument(
        "--results-dir", default="results/gru",
        help="Carpeta donde guardar split_verificacion.txt (default: results/gru)",
    )
    return parser.parse_args()


def verificar_split(splits: dict, metas: dict) -> tuple[str, list[str]]:
    """
    Construye el reporte de verificación del split y la lista de criterios que
    fallan. `splits` mapea nombre -> (X, y) sin escalar; `metas` nombre -> meta (Label).
    """
    lineas: list[str] = []
    fallos: list[str] = []

    conteo = pd.DataFrame({
        nombre: meta["Label"].value_counts() for nombre, meta in metas.items()
    }).fillna(0).astype(int)
    lineas += ["Conteo de Label por split:", conteo.to_string(), ""]

    for nombre, meta in metas.items():
        faltantes = [a for a in ATAQUES if (meta["Label"] == a).sum() == 0]
        if faltantes:
            fallos.append(f"el split '{nombre}' no tiene ejemplos de {faltantes}")

    for nombre, (X, y) in splits.items():
        for b in BLOQUES_BENIGNOS:
            n_ataque = int(y[(X["bloque"] == b).to_numpy()].sum())
            if n_ataque > 0:
                fallos.append(f"el bloque {b} de '{nombre}' tiene {n_ataque} ataques")

    lineas.append("Rango horario por split en los bloques de ataque:")
    for b in BLOQUES_ATAQUE:
        for nombre, (X, _) in splits.items():
            ts = X.loc[X["bloque"] == b, "Timestamp"]
            if ts.empty:
                lineas.append(f"  bloque {b} | {nombre:<5}: sin filas")
                fallos.append(f"el bloque {b} de '{nombre}' está vacío")
                continue
            duracion = ts.max() - ts.min()
            lineas.append(
                f"  bloque {b} | {nombre:<5}: {ts.min():%H:%M} – {ts.max():%H:%M} "
                f"({len(ts)} filas, duración {duracion})"
            )
            if nombre == "val" and duracion < DURACION_MIN_VAL:
                fallos.append(f"val dura {duracion} en el bloque {b} (< {DURACION_MIN_VAL})")
    lineas.append("")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reportar_duplicados_de_features(
            splits["train"][0], splits["val"][0], splits["test"][0],
            splits["train"][1], splits["val"][1], splits["test"][1],
        )
    lineas += ["Duplicados de features (reportar_duplicados_de_features):", buf.getvalue()]

    lineas.append("Criterios: " + ("OK" if not fallos else "FALLAN"))
    lineas += [f"  - {f}" for f in fallos]
    return "\n".join(lineas), fallos


def ataques_esperados(X: pd.DataFrame, y: pd.Series, window_size: int) -> tuple[int, int]:
    """(n_secuencias, n_ataques) que build_sequences() debe producir para este split."""
    n_seq, n_ataque = 0, 0
    for b in sorted(X["bloque"].unique()):
        y_b = y[(X["bloque"] == b).to_numpy()]
        n_seq += max(len(y_b) - window_size + 1, 0)
        n_ataque += int(y_b.iloc[window_size - 1:].sum())
    return n_seq, n_ataque


def verificar_secuencias(
    X_seq: np.ndarray, y_seq: np.ndarray, window_size: int, n_seq: int, n_ataque: int,
) -> list[str]:
    fallos = []
    esperado = (n_seq, window_size, len(FEATURE_COLUMNS))
    if X_seq.shape != esperado:
        fallos.append(f"shape X {X_seq.shape} != {esperado}")
    if y_seq.shape != (n_seq,):
        fallos.append(f"shape y {y_seq.shape} != {(n_seq,)}")
    if not np.isfinite(X_seq).all():
        fallos.append(f"X contiene {int((~np.isfinite(X_seq)).sum())} valores NaN/Inf")
    if int(y_seq.sum()) != n_ataque:
        fallos.append(f"ataques {int(y_seq.sum())} != esperado {n_ataque}")
    return fallos


def indices_objetivo(X: pd.DataFrame, window_size: int) -> np.ndarray:
    """
    Índice de X de la fila objetivo (último paso) de cada secuencia, en el mismo
    orden que build_sequences(). Es el índice que devuelve train_test_split_by_flow():
    la posición de la fila en el dataset limpio ordenado de forma estable por Timestamp.
    """
    return np.concatenate([
        X.index[(X["bloque"] == b).to_numpy()][window_size - 1:].to_numpy()
        for b in sorted(X["bloque"].unique())
    ])


def permutar_dentro_de_minuto(splits: dict, seed: int) -> dict:
    """
    Devuelve una copia de `splits` (nombre -> (X, y)) en la que, dentro de cada
    (split, bloque, Timestamp), el orden de las filas se sortea. Cada split se permuta
    por separado, así que en los minutos que cruzan una frontera la asignación de
    filas a cada split sigue siendo la del orden del archivo. X e y se reordenan con
    el mismo índice, así que cada fila conserva su etiqueta y su split. El orden por
    Timestamp se mantiene (el bloque es monótono en Timestamp), por lo que el assert
    de orden de build_sequences() sigue pasando.
    """
    rng = np.random.default_rng(seed)
    permutados = {}
    for nombre, (X, y) in splits.items():
        clave = rng.random(len(X))
        # np.lexsort ordena por la última clave primero: bloque, Timestamp, sorteo
        orden = np.lexsort((clave, X["Timestamp"].to_numpy(), X["bloque"].to_numpy()))
        permutados[nombre] = (X.iloc[orden], y.iloc[orden])
    return permutados


def verificar_permutacion(splits_csv: dict, splits_perm: dict) -> list[str]:
    """
    Comprueba que la variante permutada tiene exactamente las mismas filas, con las
    mismas etiquetas y features, en cada split y bloque que la variante "csv", y que
    sigue ordenada por Timestamp.
    """
    fallos = []
    for nombre, (X_csv, y_csv) in splits_csv.items():
        X_p, y_p = splits_perm[nombre]
        if not X_p["Timestamp"].is_monotonic_increasing:
            fallos.append(f"'{nombre}': el orden por Timestamp no se conservó")
        if not X_p.sort_index().equals(X_csv.sort_index()):
            fallos.append(f"'{nombre}': las filas de X no son las mismas que en 'csv'")
        if not y_p.sort_index().equals(y_csv.sort_index()):
            fallos.append(f"'{nombre}': las etiquetas no son las mismas que en 'csv'")
        atk_csv = y_csv.groupby(X_csv["bloque"]).sum()
        atk_p = y_p.groupby(X_p["bloque"]).sum()
        if not atk_p.equals(atk_csv):
            fallos.append(
                f"'{nombre}': ataques por bloque distintos (csv {atk_csv.to_dict()}, "
                f"aleatorio {atk_p.to_dict()})"
            )
    return fallos


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Cargar", args.processed_path)
    print("=" * 70)
    df = cargar_dataset_limpio(args.processed_path)

    print()
    print("=" * 70)
    print("Split cronológico por bloques (desempate='csv')")
    print("=" * 70)
    (X_train, X_val, X_test, y_train, y_val, y_test,
     meta_train, meta_val, meta_test) = train_test_split_by_flow(
        df, desempate="csv", return_meta=True,
    )

    columnas_x = set(X_train.columns) - {"Timestamp", "bloque"}
    if columnas_x != set(FEATURE_COLUMNS):
        raise RuntimeError(
            "Las columnas de X no coinciden con FEATURE_COLUMNS: "
            f"sobran {sorted(columnas_x - set(FEATURE_COLUMNS))}, "
            f"faltan {sorted(set(FEATURE_COLUMNS) - columnas_x)}"
        )

    reporte, fallos = verificar_split(
        {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)},
        {"train": meta_train, "val": meta_val, "test": meta_test},
    )
    print()
    print(reporte)
    results_dir.mkdir(parents=True, exist_ok=True)
    reporte_path = results_dir / "split_verificacion.txt"
    reporte_path.write_text(reporte + "\n", encoding="utf-8")
    print(f"Reporte guardado en {reporte_path}")
    if fallos:
        raise RuntimeError(
            "El split no cumple los criterios de verificación; no se generó "
            f"ningún .npy. Ver {reporte_path}:\n  - " + "\n  - ".join(fallos)
        )

    print()
    print("=" * 70)
    print(f"Escalado (fit solo en train, transformacion_log={TRANSFORMACION_LOG})")
    print("=" * 70)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(
        X_train, X_val, X_test, transformacion_log=TRANSFORMACION_LOG,
    )
    # Un único scaler: la variante "aleatorio" reutiliza las mismas filas ya escaladas
    scaler_path = output_dir / "scaler_csv.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler guardado en {scaler_path}")

    splits_csv = {
        "train": (X_train_s, y_train),
        "val": (X_val_s, y_val),
        "test": (X_test_s, y_test),
    }

    # (variante, split, window_size, shape_X, n_ataque, n_benigno)
    resumen: list[tuple[str, str, int, tuple[int, ...], int, int]] = []
    guardados: list[Path] = []
    # (variante, split, window_size) -> ataques en las secuencias
    ataques_seq: dict[tuple[str, str, int], int] = {}

    for variante, window_sizes in VARIANTES.items():
        if variante == "csv":
            splits = splits_csv
        else:
            print()
            print("=" * 70)
            print(f"Variante {variante!r}: permutación dentro de cada (split, bloque, Timestamp)")
            print("=" * 70)
            splits = permutar_dentro_de_minuto(splits_csv, SEMILLA_PERMUTACION)
            fallos = verificar_permutacion(splits_csv, splits)
            if fallos:
                raise RuntimeError(
                    f"La variante {variante!r} no conserva las filas de 'csv'; no se "
                    "guardó ningún .npy de esta variante:\n  - " + "\n  - ".join(fallos)
                )
            for nombre, (_, y) in splits.items():
                print(f"  {nombre}: {int(y.sum())} ataques por fila (idénticos a 'csv')")
        sufijo = "" if variante == "csv" else f"_{variante}"

        for window_size in window_sizes:
            print()
            print("=" * 70)
            print(f"Generando secuencias -- variante={variante}, window_size={window_size}")
            print("=" * 70)
            for split_name, (X, y) in splits.items():
                X_seq, y_seq = build_sequences(X, y, window_size)
                n_seq, n_ataque = ataques_esperados(X, y, window_size)
                fallos = verificar_secuencias(X_seq, y_seq, window_size, n_seq, n_ataque)
                if fallos:
                    raise RuntimeError(
                        f"Secuencias inválidas ({variante}, {split_name}, W={window_size}); "
                        "no se guardaron:\n  - " + "\n  - ".join(fallos)
                    )
                idx_seq = indices_objetivo(X, window_size)
                if len(idx_seq) != len(y_seq) or not np.array_equal(y.loc[idx_seq].to_numpy(), y_seq):
                    raise RuntimeError(
                        f"Índices objetivo inconsistentes con y ({variante}, {split_name}, "
                        f"W={window_size}): len(idx)={len(idx_seq)}, len(y)={len(y_seq)}"
                    )

                x_path = output_dir / f"X_{split_name}_w{window_size}{sufijo}.npy"
                y_path = output_dir / f"y_{split_name}_w{window_size}{sufijo}.npy"
                idx_path = output_dir / f"idx_{split_name}_w{window_size}{sufijo}.npy"
                np.save(x_path, X_seq)
                np.save(y_path, y_seq)
                np.save(idx_path, idx_seq)
                guardados += [x_path, y_path, idx_path]
                print(f"  {split_name}: X{X_seq.shape} -> {x_path}")
                print(f"  {split_name}: y{y_seq.shape} -> {y_path}")
                print(f"  {split_name}: idx{idx_seq.shape} -> {idx_path}")

                n_ataque_seq = int(y_seq.sum())
                ataques_seq[(variante, split_name, window_size)] = n_ataque_seq
                ref = ataques_seq.get(("csv", split_name, window_size))
                if variante != "csv" and ref is not None:
                    # Puede diferir levemente: cambia qué filas caen en las primeras
                    # W-1 posiciones de cada bloque, que no generan secuencia
                    print(
                        f"  {split_name}: ataques en secuencias {n_ataque_seq} "
                        f"vs 'csv' {ref} (diferencia {n_ataque_seq - ref:+d})"
                    )

                resumen.append((
                    variante, split_name, window_size, X_seq.shape,
                    int((y_seq == 1).sum()), int((y_seq == 0).sum()),
                ))
                del X_seq, y_seq

    print()
    print("=" * 70)
    print("Resumen")
    print("=" * 70)
    header = f"{'variante':<10} {'split':<6} {'W':>3}  {'shape X':<22} {'ataque':>8} {'benigno':>9}"
    print(header)
    print("-" * len(header))
    for variante, split_name, window_size, shape, n_ataque, n_benigno in resumen:
        print(
            f"{variante:<10} {split_name:<6} {window_size:>3}  {str(shape):<22} "
            f"{n_ataque:>8} {n_benigno:>9}"
        )

    print()
    print(
        f"Verificación OK -- {len(guardados)} archivos .npy y 1 scaler "
        f"({scaler_path.name}) generados en {output_dir}/"
    )


if __name__ == "__main__":
    main()
