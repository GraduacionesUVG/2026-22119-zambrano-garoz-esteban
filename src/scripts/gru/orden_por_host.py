"""
Registra en results/gru/00_datos/orden_por_host.json cuánto agrupa por host el orden
del CSV crudo del miércoles: entre filas consecutivas del mismo minuto, qué proporción
comparte Source IP y Destination IP, frente a la esperada si el orden dentro de cada
minuto fuera aleatorio (ADR 0016).

Usa la captura filtrada por clase (load_raw_csvs + filter_low_and_slow), porque el CSV
procesado no conserva las IPs. El orden es el del archivo y el minuto es el Timestamp
corregido con parsear_timestamp().

El bloque "orden_de_las_ventanas" repite la medición en el orden en que el GRU arma sus
ventanas: recupera las IPs de las filas del CSV procesado por su índice en el CSV crudo
(ver orden_de_las_ventanas()) y aplica el orden de train_test_split_by_flow().

Uso:
    python scripts/gru/orden_por_host.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # no dejar __pycache__ en el repo al importar src

import argparse  # noqa: E402
import contextlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.models.gru.sequences import FEATURE_COLUMNS  # noqa: E402
from src.preprocessing.clean import (  # noqa: E402
    basic_cleaning,
    cargar_dataset_limpio,
    encode_labels,
    excluir_ataques_fuera_de_ventana,
    excluir_respuestas_de_ataque,
    filter_low_and_slow,
    load_raw_csvs,
    parsear_timestamp,
    train_test_split_by_flow,
)

# permutar_dentro_de_minuto() y su semilla, tal como los usa la variante "aleatorio"
# (cargado por ruta: scripts/ no es un paquete)
_spec = importlib.util.spec_from_file_location("build_gru_sequences", RAIZ / "scripts" / "gru" / "build_gru_sequences.py")
_secuencias = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_secuencias)

FILAS_ESPERADAS = 451_326  # tras el filtro de clases (results/gru/00_datos/conteos_limpieza.json)
COLUMNAS_IP = ("Source IP", "Destination IP")

DEFINICIONES = {
    "par_consecutivo_mismo_minuto": (
        "par de filas i-1, i consecutivas en el orden del archivo (tras el filtro de clases) "
        "cuyo Timestamp corregido es el mismo minuto"
    ),
    "pct_observado": (
        "100 · (pares consecutivos del mismo minuto con la misma IP) / "
        "(pares consecutivos del mismo minuto)"
    ),
    "pct_esperado_azar": (
        "100 · Σ_m pares_m · Σ_ip p_{m,ip}² / Σ_m pares_m, donde m recorre los minutos, "
        "pares_m es el número de pares consecutivos del mismo minuto que caen en m y "
        "p_{m,ip} es la proporción de filas del minuto m con esa IP. Σ_ip p² es la "
        "probabilidad de que dos filas del minuto tomadas al azar compartan IP"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-path", default="data/processed/wednesday_low_and_slow_clean.csv")
    parser.add_argument("--output-dir", default="results/gru/00_datos")
    return parser.parse_args()


def concentracion_por_minuto(minuto: pd.Series, ip: pd.Series) -> pd.Series:
    """Σ_ip p_{m,ip}² por minuto."""
    conteos = pd.DataFrame({"minuto": minuto.to_numpy(), "ip": ip.to_numpy()}).groupby(
        ["minuto", "ip"], dropna=False).size()
    n = conteos.groupby(level="minuto").sum()
    return (conteos ** 2).groupby(level="minuto").sum() / n ** 2


def coincidencia(minuto: pd.Series, ip: pd.Series, ponderar_por: str) -> dict:
    """
    Observado y esperado al azar para filas consecutivas del mismo minuto.
    ponderar_por="pares" usa pares_m (definición de este script); "filas" usa el número
    de filas del minuto (cálculo de la sección 5 de notebooks/gru/02_ablaciones.ipynb).
    """
    mismo_minuto = (minuto == minuto.shift()).to_numpy()
    misma_ip = (ip == ip.shift()).to_numpy()
    pares = int(mismo_minuto.sum())
    pares_misma_ip = int((mismo_minuto & misma_ip).sum())

    p2 = concentracion_por_minuto(minuto, ip)
    if ponderar_por == "pares":
        peso = pd.Series(minuto.to_numpy()[mismo_minuto]).value_counts()
    else:
        peso = pd.Series(minuto.to_numpy()).value_counts()
    peso = peso.reindex(p2.index, fill_value=0)
    return {
        "pares_mismo_minuto": pares,
        "pares_misma_ip": pares_misma_ip,
        "pct_observado": 100 * pares_misma_ip / pares,
        "pct_esperado_azar": 100 * float((p2 * peso).sum() / peso.sum()),
    }


def coincidencia_por_celdas(minuto: np.ndarray, ip: np.ndarray, grupo: np.ndarray) -> dict:
    """
    Como coincidencia(..., "pares"), pero un par cuenta solo si sus dos filas comparten
    `grupo` (partición y bloque, que las ventanas no cruzan), y el esperado al azar usa
    Σ_ip p² de cada celda (grupo, minuto), que es donde permuta la variante aleatoria.
    """
    mismo = (minuto[1:] == minuto[:-1]) & (grupo[1:] == grupo[:-1])
    misma_ip = ip[1:] == ip[:-1]
    pares = int(mismo.sum())
    pares_misma_ip = int((mismo & misma_ip).sum())

    celdas = ["grupo", "minuto"]
    conteos = pd.DataFrame({"grupo": grupo, "minuto": minuto, "ip": ip}).groupby(
        [*celdas, "ip"], dropna=False).size()
    n = conteos.groupby(level=celdas).sum()
    p2 = (conteos ** 2).groupby(level=celdas).sum() / n ** 2
    peso = pd.DataFrame({"grupo": grupo[1:][mismo], "minuto": minuto[1:][mismo]}).groupby(celdas).size()
    peso = peso.reindex(p2.index, fill_value=0)
    return {
        "pares_mismo_minuto": pares,
        "pares_misma_ip": pares_misma_ip,
        "pct_observado": 100 * pares_misma_ip / pares,
        "pct_esperado_azar": 100 * float((p2 * peso).sum() / peso.sum()),
    }


def mapear_ips(filtrado_indice_crudo: pd.DataFrame, processed_path: str) -> tuple[pd.DataFrame | None, list[str]]:
    """
    CSV procesado con las IPs de cada fila. El CSV se guardó con index=False
    (notebooks/Cleaning/limpieza_low_and_slow.ipynb), pero ningún paso de load_and_clean()
    reinicia el índice: se repiten los pasos posteriores al filtro de clases sobre la
    captura con su índice del CSV crudo y se comprueba que el resultado coincide fila a
    fila con el CSV procesado. Si coincide, la fila k del CSV es la fila de índice
    limpio.index[k] del crudo. Devuelve (None, fallos) si no coincide.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        limpio = excluir_respuestas_de_ataque(filtrado_indice_crudo)
        limpio = basic_cleaning(limpio)
        limpio = encode_labels(limpio)
        limpio = excluir_ataques_fuera_de_ventana(limpio)
    procesado = cargar_dataset_limpio(processed_path)

    fallos = []
    if len(limpio) != len(procesado):
        return None, [f"filas: recálculo {len(limpio):,}, CSV procesado {len(procesado):,}"]
    if list(limpio.columns) != list(procesado.columns):
        fallos.append("columnas distintas o en otro orden")
    if not np.array_equal(limpio["Label"].to_numpy(dtype=object), procesado["Label"].to_numpy(dtype=object)):
        fallos.append("Label distinta en alguna fila")
    if not np.array_equal(limpio["label_binaria"].to_numpy(), procesado["label_binaria"].to_numpy()):
        fallos.append("label_binaria distinta en alguna fila")
    if not np.array_equal(limpio["Timestamp"].to_numpy().astype("datetime64[s]"),
                          procesado["Timestamp"].to_numpy().astype("datetime64[s]")):
        fallos.append("Timestamp distinto en alguna fila")
    if not np.allclose(limpio[FEATURE_COLUMNS].to_numpy(float), procesado[FEATURE_COLUMNS].to_numpy(float),
                       rtol=1e-12, atol=0, equal_nan=True):
        fallos.append("features distintas en alguna fila (rtol 1e-12)")
    if fallos:
        return None, fallos

    for col in COLUMNAS_IP:
        procesado[col] = filtrado_indice_crudo.loc[limpio.index, col].to_numpy(dtype=object)
    procesado["indice_crudo"] = limpio.index.to_numpy()
    return procesado, []


def _medir_orden(splits: dict) -> dict:
    """splits: nombre -> (X, y) en el orden de las ventanas; pares dentro de (partición, bloque)."""
    minuto, grupo, ips = [], [], {col: [] for col in COLUMNAS_IP}
    for nombre in ("train", "val", "test"):
        X = splits[nombre][0]
        minuto.append(X["Timestamp"].dt.floor("min").to_numpy())
        grupo.append((nombre + "|" + X["bloque"].astype(str)).to_numpy(dtype=object))
        for col in COLUMNAS_IP:
            ips[col].append(X[col].to_numpy(dtype=object))
    minuto, grupo = np.concatenate(minuto), np.concatenate(grupo)
    return {col: coincidencia_por_celdas(minuto, np.concatenate(ips[col]), grupo) for col in COLUMNAS_IP}


def orden_de_las_ventanas(filtrado_indice_crudo: pd.DataFrame, filtrado: pd.DataFrame,
                          timestamp: pd.Series, processed_path: str) -> dict:
    definicion_par = (
        "par de filas consecutivas en el orden indicado que pertenecen a la misma partición y "
        "al mismo bloque (las ventanas del GRU no cruzan ninguno de los dos) y tienen el mismo "
        "minuto. El esperado al azar usa Σ_ip p² de cada celda (partición, bloque, minuto), "
        "ponderado por los pares de la celda."
    )
    procesado, fallos = mapear_ips(filtrado_indice_crudo, processed_path)
    if procesado is None:
        orden = np.argsort(timestamp.to_numpy(), kind="stable")
        minuto = timestamp.dt.floor("min").to_numpy()[orden]
        grupo = np.zeros(len(orden), dtype=int)
        return {
            "descripcion": (
                "No se pudo mapear el CSV procesado al crudo (" + "; ".join(fallos) + "). Se usa la "
                f"captura filtrada por clase ({len(filtrado):,} filas) ordenada de forma estable por "
                "el Timestamp corregido, sin partición ni bloque; no es exactamente el orden del GRU."
            ),
            "definicion_par": "par consecutivo del mismo minuto en ese orden; esperado por pares como en el bloque principal",
            "por_columna": {col: coincidencia_por_celdas(minuto, filtrado[col].to_numpy(dtype=object)[orden], grupo)
                            for col in COLUMNAS_IP},
        }

    with contextlib.redirect_stdout(io.StringIO()):
        salida = train_test_split_by_flow(procesado, return_meta=True)
    splits = {nombre: (salida[i], salida[i + 3]) for i, nombre in enumerate(("train", "val", "test"))}
    for nombre, (X, _) in splits.items():
        if not (np.diff(X.index.to_numpy()) > 0).all():
            raise RuntimeError(f"X_{nombre} no está en el orden del índice de train_test_split_by_flow()")
    permutados = _secuencias.permutar_dentro_de_minuto(splits, _secuencias.SEMILLA_PERMUTACION)
    return {
        "descripcion": (
            f"Filas del CSV procesado ({len(procesado):,}) con sus IPs del CSV crudo, en el orden de "
            "train_test_split_by_flow() con sus valores por defecto: sort_values('Timestamp', "
            "kind='stable') (src/preprocessing/clean.py:560), que conserva el orden del CSV procesado "
            "en los empates de minuto, y corte por bloque y posición 70/15/15. Es el orden de la "
            "variante csv de las secuencias."
        ),
        "mapeo": {
            "metodo": (
                "load_and_clean() no reinicia el índice (filter_low_and_slow, excluir_respuestas_de_ataque, "
                "basic_cleaning y excluir_ataques_fuera_de_ventana filtran con máscaras), así que el índice "
                "de cada fila limpia es su fila en el CSV crudo (RangeIndex de load_raw_csvs). Se repiten "
                "esos pasos y se compara fila a fila con el CSV procesado, guardado en el mismo orden."
            ),
            "verificacion": "filas, columnas, Label, label_binaria, Timestamp y 59 features (rtol 1e-12): OK",
            "filas": int(len(procesado)),
        },
        "definicion_par": definicion_par,
        "por_columna": _medir_orden(splits),
        "variante_aleatoria": {
            "descripcion": (
                "Mismo cálculo en el orden de la variante aleatorio: permutar_dentro_de_minuto() de "
                "scripts/gru/build_gru_sequences.py con SEMILLA_PERMUTACION, que sortea el orden dentro de "
                "cada (partición, bloque, Timestamp). Contraste: el esperado al azar es el mismo."
            ),
            "semilla": _secuencias.SEMILLA_PERMUTACION,
            "por_columna": _medir_orden(permutados),
        },
    }


def main() -> None:
    args = parse_args()
    salida_carga = io.StringIO()
    with contextlib.redirect_stdout(salida_carga):
        crudo = load_raw_csvs(args.raw_dir)
        filtrado_indice_crudo = filter_low_and_slow(crudo)
        filtrado = filtrado_indice_crudo.reset_index(drop=True)
        timestamp = parsear_timestamp(filtrado["Timestamp"])
    del crudo
    archivo = re.search(r"Cargado: (.+)", salida_carga.getvalue())
    print(salida_carga.getvalue())

    if len(filtrado) != FILAS_ESPERADAS:
        raise RuntimeError(f"El filtro de clases deja {len(filtrado):,} filas; se esperaban {FILAS_ESPERADAS:,}")
    if timestamp.isna().any():
        raise RuntimeError(f"{int(timestamp.isna().sum())} Timestamp no parseables")

    minuto = timestamp.dt.floor("min")
    minuto_texto = filtrado["Timestamp"].astype(str).str.strip()
    resultado = {
        "fuente": {
            "archivo": archivo.group(1).strip() if archivo else None,
            "filtro": "filter_low_and_slow(): BENIGN, DoS slowloris y DoS Slowhttptest; orden del archivo",
            "correccion_timestamp": "parsear_timestamp(): horas < 8 se desplazan +12 h",
            "filas": int(len(filtrado)),
            "filas_por_clase": {k: int(v) for k, v in filtrado["Label"].value_counts().items()},
            "minutos": int(minuto.nunique()),
            "pares_consecutivos": int(len(filtrado) - 1),
        },
        "definiciones": DEFINICIONES,
        "por_columna": {col: coincidencia(minuto, filtrado[col], "pares") for col in COLUMNAS_IP},
        "reconciliacion_02_ablaciones": {
            "descripcion": (
                "Mismo cálculo que la sección 5 de notebooks/gru/02_ablaciones.ipynb: minuto = texto "
                "crudo del Timestamp (sin corregir) y esperado ponderado por el número de filas de "
                "cada minuto en lugar de por sus pares consecutivos."
            ),
            "por_columna": {col: coincidencia(minuto_texto, filtrado[col], "filas") for col in COLUMNAS_IP},
        },
    }
    for col, d in resultado["por_columna"].items():
        if d["pares_mismo_minuto"] == 0 or np.isnan(d["pct_esperado_azar"]):
            raise RuntimeError(f"{col}: no hay pares consecutivos del mismo minuto")

    print("Orden de las ventanas: repitiendo la limpieza y aplicando train_test_split_by_flow()...", flush=True)
    resultado["orden_de_las_ventanas"] = orden_de_las_ventanas(
        filtrado_indice_crudo, filtrado, timestamp, str(RAIZ / args.processed_path))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "orden_por_host.json"
    path.write_text(json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{path}:\n{json.dumps(resultado, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
