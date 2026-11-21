from __future__ import annotations

import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

# ============================================================
# CONFIGURACION GENERAL
# ============================================================

DATA_DIR = ROOT / "Data_Test"
MODEL_DIR = ROOT / "Models"

GRU_DATA_DIR = DATA_DIR / "gru_test"

# Carpeta generada por src/scripts/gru/exportar_demo.py de este repositorio
GRU_MODEL_DIR = MODEL_DIR / "GRU"

NOMBRES_BLOQUE = {
    1: "Bloque 1 · antes de Slowloris",
    2: "Bloque 2 · Slowloris",
    3: "Bloque 3 · entre ataques",
    4: "Bloque 4 · Slowhttptest",
    5: "Bloque 5 · después de Slowhttptest",
}

COLUMNAS_INSPECCION = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Init_Win_bytes_forward",
]

st.set_page_config(
    page_title="Low & Slow IDS | Live Demo",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        max-width: 1450px;
    }

    [data-testid="stMetric"] {
        background: rgba(127,127,127,.07);
        padding: 12px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def csv_from_upload(file) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file.getvalue()))


# ============================================================
# CARGA DE MODELOS
# ============================================================

@st.cache_data(show_spinner=False)
def load_gru_config(model_dir: str) -> dict:
    from src.models.gru.demo import cargar_config

    return cargar_config(model_dir)


@st.cache_resource(show_spinner=False)
def load_gru_model(model_dir: str, archivo: str):
    from src.models.gru.demo import cargar_modelo

    return cargar_modelo(Path(model_dir) / archivo)


@st.cache_resource(show_spinner=False)
def load_gru_scaler(model_dir: str):
    from src.models.gru.demo import cargar_scaler

    return cargar_scaler(model_dir)


# ============================================================
# ETIQUETAS
# ============================================================

def extract_labels(y_frame: pd.DataFrame) -> np.ndarray:

    if y_frame.shape[1] != 1:
        raise ValueError(
            "y_test.csv debe contener una sola columna."
        )

    y = y_frame.iloc[:, 0]

    if y.isna().any():
        raise ValueError(
            "Las etiquetas contienen valores nulos."
        )

    if pd.api.types.is_numeric_dtype(y):

        values = y.to_numpy()

        if not np.isin(values, [0, 1]).all():
            raise ValueError(
                "Las etiquetas deben ser binarias (0/1)."
            )

        return values.astype(int)

    normalized = y.astype(str).str.strip().str.lower()

    mapping = {
        "0": 0,
        "benign": 0,
        "benigno": 0,
        "normal": 0,
        "1": 1,
        "attack": 1,
        "ataque": 1,
        "slowloris": 1,
        "slowhttptest": 1,
        "slowhttp": 1,
    }

    if not normalized.isin(mapping).all():

        bad = normalized[
            ~normalized.isin(mapping)
        ].unique()[:5]

        raise ValueError(
            f"Etiquetas no reconocidas: {bad.tolist()}"
        )

    return normalized.map(mapping).to_numpy(dtype=int)


# ============================================================
# PREDICCIONES GRU
# ============================================================

def prepare_gru(
    x_df: pd.DataFrame,
    model_dir: str,
    config: dict,
    meta: pd.DataFrame | None,
    solo_comunes: bool,
):
    """
    Pipeline de la tesis: log con signo + scaler de train, ventanas de W flujos
    por bloque. Devuelve las ventanas y la fila objetivo de cada una.
    """

    from src.models.gru.demo import (
        construir_ventanas,
        escalar_flujos,
    )

    if "bloque" not in x_df.columns:
        raise ValueError(
            "X_test de la GRU debe incluir la columna 'bloque' "
            "(las ventanas no cruzan bloques)."
        )

    escalado = escalar_flujos(
        x_df,
        load_gru_scaler(model_dir),
        config["caracteristicas"]
    )

    ventanas, objetivos = construir_ventanas(
        escalado,
        x_df["bloque"].to_numpy(),
        int(config["ventana"])
    )

    if solo_comunes and meta is not None:

        mask = meta["en_comun"].to_numpy(dtype=bool)[objetivos]
        ventanas = ventanas[mask]
        objetivos = objetivos[mask]

    return ventanas, objetivos


def predict_gru(
    ventanas: np.ndarray,
    model_dir: str,
    config: dict
):

    from src.models.gru.demo import predecir

    model = load_gru_model(
        model_dir,
        config["archivo"]
    )

    scores = predecir(model, ventanas)

    if (
        not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError(
            "La GRU no devolvió probabilidades válidas."
        )

    predictions = (
        scores >= float(config["umbral"])
    ).astype(int)

    return predictions, scores


# ============================================================
# METRICAS
# ============================================================

def evaluate(y_true, y_pred, scores):

    result = {
        "Accuracy": accuracy_score(
            y_true, y_pred
        ),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_true, y_pred, zero_division=0
        ),
    }

    if len(np.unique(y_true)) == 2:

        result["ROC-AUC"] = roc_auc_score(
            y_true, scores
        )

        result["AP"] = average_precision_score(
            y_true, scores
        )

    else:

        result["ROC-AUC"] = float("nan")
        result["AP"] = float("nan")

    return result


# ============================================================
# MATRIZ DE CONFUSION
# ============================================================

def draw_confusion(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(
        figsize=(5.5, 4)
    )

    ax.imshow(
        cm,
        cmap="Blues"
    )

    for (i, j), value in np.ndenumerate(cm):

        ax.text(
            j,
            i,
            f"{value:,}",
            ha="center",
            va="center",
            fontsize=14,
            color=(
                "white"
                if value > cm.max() / 2
                else "black"
            )
        )

    ax.set_xticks(
        [0, 1],
        ["Benigno", "Ataque"]
    )

    ax.set_yticks(
        [0, 1],
        ["Benigno", "Ataque"]
    )

    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de confusion")

    fig.tight_layout()

    return fig, cm


# ============================================================
# GRU: SECCION ESPECIFICA
# ============================================================

def draw_gru_timeline(
    block: pd.DataFrame,
    threshold: float
):

    y = block["etiqueta_real"].to_numpy()
    p = block["prediccion"].to_numpy()
    s = block["score"].to_numpy()
    x = np.arange(len(block))

    fig, ax = plt.subplots(figsize=(11, 3.6))

    groups = [
        ((y == 0) & (p == 0), "Benigno", "#9a9a9a", "o", 4, 0.35),
        ((y == 1) & (p == 1), "Ataque detectado", "#2a78d6", "o", 9, 0.8),
        ((y == 0) & (p == 1), "Falso positivo", "#eb6834", "^", 28, 0.9),
        ((y == 1) & (p == 0), "Falso negativo", "#d62728", "X", 48, 1.0),
    ]

    for mask, label, color, marker, size, alpha in groups:

        if mask.any():
            ax.scatter(
                x[mask],
                s[mask],
                s=size,
                c=color,
                marker=marker,
                alpha=alpha,
                linewidths=0,
                label=f"{label} ({int(mask.sum()):,})"
            )

    ax.axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Umbral = {threshold:.3f}"
    )

    if "Timestamp" in block.columns and len(block) > 1:

        times = pd.to_datetime(block["Timestamp"].to_numpy())

        # Un tick en el primer flujo de cada minuto (máximo ~8)
        ticks = np.flatnonzero(
            np.r_[True, times[1:] != times[:-1]]
        )
        ticks = ticks[::max(1, int(np.ceil(len(ticks) / 8)))]

        ax.set_xticks(ticks, [f"{t:%H:%M}" for t in times[ticks]])
        ax.set_xlabel("Minuto del flujo objetivo")

    else:

        ax.set_xlabel("Flujo objetivo (orden de las ventanas)")

    ax.set_ylabel("P(ataque)")
    ax.set_ylim(-0.03, 1.03)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=9
    )

    fig.tight_layout()

    return fig


def show_gru_section(data, y_true, pred, scores):

    gru = data["gru"]
    config = gru["config"]
    seed = config["semilla"]
    result = data["predictions"]
    threshold = data["threshold"]
    window = int(config["ventana"])

    st.subheader("GRU · análisis del modelo principal")

    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    # --------------------------------------------------------
    # Verificacion contra la tesis
    # --------------------------------------------------------

    if gru["solo_comunes"]:

        expected = config["test_esperado"]
        obtained = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

        diffs = [
            f"{k.upper()}: {int(obtained[k]):,} (tesis: {int(expected[k]):,})"
            for k in obtained
            if int(obtained[k]) != int(expected[k])
        ]

        if not diffs:

            st.success(
                f"Reproduce la evaluación en test de la tesis "
                f"(semilla {seed}): TP {tp:,} · FP {fp:,} · "
                f"FN {fn:,} · TN {tn:,}."
            )

        else:

            st.warning(
                "No coincide con test_resultados.json → "
                + " · ".join(diffs)
            )

    else:

        st.caption(
            "Se evalúan todas las filas con ventana. La tesis "
            "reporta solo las filas comunes de su evaluación."
        )

    # --------------------------------------------------------
    # Recall por tipo de ataque
    # --------------------------------------------------------

    if "tipo" in result.columns:

        types = {
            "DoS slowloris": "Slowloris",
            "DoS Slowhttptest": "Slowhttptest",
        }

        present = [
            t for t in types
            if (result["tipo"] == t).any()
        ]

        cols = st.columns(len(present) + 1)

        for col, t in zip(cols, present):

            mask = (result["tipo"] == t).to_numpy()
            detected = int(pred[mask].sum())
            total = int(mask.sum())

            col.metric(
                f"Recall {types[t]}",
                f"{detected / total:.2%}"
            )

            col.caption(
                f"{total - detected} de {total:,} ataques sin detectar"
            )

        cols[-1].metric(
            "Falsos positivos",
            f"{fp:,}"
        )

        cols[-1].caption(
            f"{fp / max(fp + tn, 1):.2%} del tráfico benigno"
        )

    # --------------------------------------------------------
    # Linea de tiempo
    # --------------------------------------------------------

    st.markdown("**Probabilidad de ataque a lo largo del test**")

    blocks = sorted(
        int(b) for b in result["bloque"].unique()
    )

    default_block = 2 if 2 in blocks else blocks[0]

    block_id = st.selectbox(
        "Bloque temporal",
        blocks,
        index=blocks.index(default_block),
        format_func=lambda b: NOMBRES_BLOQUE.get(b, f"Bloque {b}"),
        key="gru_bloque"
    )

    fig = draw_gru_timeline(
        result[result["bloque"] == block_id],
        threshold
    )

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------------
    # Inspeccion de una ventana
    # --------------------------------------------------------

    st.markdown(
        f"**Inspeccionar una ventana** · la GRU recibe "
        f"{window} flujos consecutivos y clasifica el último"
    )

    cases = {
        "Falsos negativos": (y_true == 1) & (pred == 0),
        "Falsos positivos": (y_true == 0) & (pred == 1),
        "Ataques detectados": (y_true == 1) & (pred == 1),
        "Benignos correctos": (y_true == 0) & (pred == 0),
    }

    c1, c2 = st.columns([3, 1])

    case = c1.radio(
        "Caso",
        list(cases),
        format_func=lambda k: f"{k} ({int(cases[k].sum()):,})",
        horizontal=True,
        key="gru_caso"
    )

    positions = np.flatnonzero(cases[case])

    if len(positions) == 0:

        st.info("No hay casos de este tipo.")

    else:

        k = c2.number_input(
            "Número de caso",
            min_value=1,
            max_value=len(positions),
            value=1,
            key=f"gru_num_{case}"
        )

        i = positions[int(k) - 1]
        target = int(result["fila_original"].iloc[i])
        rows = np.arange(target - window + 1, target + 1)

        x_df = gru["x_df"]
        meta = gru["meta"]

        columns = [
            c for c in COLUMNAS_INSPECCION
            if c in x_df.columns
        ]

        table = x_df.iloc[rows][columns].reset_index(drop=True)

        table.insert(
            0,
            "Paso",
            [
                f"t-{window - 1 - j}" if j < window - 1
                else "t (obj.)"
                for j in range(window)
            ]
        )

        if meta is not None:

            table.insert(
                1,
                "Etiqueta",
                meta["Label"].to_numpy()[rows]
            )

            table.insert(
                1,
                "Timestamp",
                meta["Timestamp"].to_numpy()[rows]
            )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )

        verdict = "Ataque" if pred[i] == 1 else "Benigno"
        real = "Ataque" if y_true[i] == 1 else "Benigno"

        st.caption(
            f"P(ataque) = {scores[i]:.4f} · umbral {threshold:.3f} "
            f"→ predicción: **{verdict}** · etiqueta real: **{real}**. "
            "Valores sin escalar; el modelo los recibe con log "
            "con signo y StandardScaler."
        )

    # --------------------------------------------------------
    # Referencia: media de las semillas de la tesis
    # --------------------------------------------------------

    if gru["solo_comunes"]:

        ref_mean = config["test_media"]
        ref_std = config["test_std"]

        reference = " · ".join(
            f"{name} {ref_mean[k]:.3f} ± {ref_std[k]:.3f}"
            for name, k in (
                ("Recall", "recall"),
                ("Precision", "precision"),
                ("F1-score", "f1"),
                ("PR-AUC", "pr_auc"),
            )
        )

        st.caption(
            f"La tesis reporta la media ± desv. estándar de "
            f"{config['n_semillas_tesis']} semillas: {reference}. "
            f"La demo usa la semilla {seed}, la más cercana a esa media."
        )


# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

st.title("Detección de ataques Low & Slow")

st.caption(
    "Demostración interactiva de inferencia sobre "
    "flujos de red · GRU"
)


# ============================================================
# BARRA LATERAL
# ============================================================

with st.sidebar:

    st.header("Configuración")

    model_path = st.text_input(
        "Carpeta del modelo",
        str(GRU_MODEL_DIR)
    )

    st.divider()

    source = st.radio(
        "Origen de datos",
        ["Archivos locales", "Subir CSV"]
    )

    if source == "Archivos locales":

        x_path = st.text_input(
            "Ruta X_test.csv",
            str(GRU_DATA_DIR / "X_test.csv"),
            key="x_path_GRU"
        )

        y_path = st.text_input(
            "Ruta y_test.csv",
            str(GRU_DATA_DIR / "y_test.csv"),
            key="y_path_GRU"
        )

        meta_path = st.text_input(
            "Ruta meta_test.csv (opcional)",
            str(GRU_DATA_DIR / "meta_test.csv"),
            key="meta_path_GRU"
        )

        x_upload = None
        y_upload = None
        meta_upload = None

    else:

        x_upload = st.file_uploader(
            "X_test.csv",
            type="csv",
            key="x"
        )

        y_upload = st.file_uploader(
            "y_test.csv (opcional)",
            type="csv",
            key="y"
        )

        meta_upload = st.file_uploader(
            "meta_test.csv (opcional)",
            type="csv",
            key="meta"
        )

        x_path = ""
        y_path = ""
        meta_path = ""

    st.divider()

    try:
        gru_config = load_gru_config(model_path)
    except Exception as exc:
        gru_config = None
        st.error(f"{exc}")

    if gru_config is not None:

        threshold = float(gru_config["umbral"])

        st.success(
            f"**{gru_config['nombre']}**  \n"
            f"Semilla {gru_config['semilla']}  \n"
            f"Ventana: {gru_config['ventana']} flujos  \n"
            f"Umbral F2 (validación): {threshold:.3f}  \n"
            "Log con signo + StandardScaler de train."
        )

    solo_comunes = st.checkbox(
        "Solo filas comunes de la evaluación de la tesis",
        value=True,
        help=(
            "Evalúa las mismas filas objetivo que la tesis "
            "(idx_test_comun). Las demás filas se usan solo "
            "como historia de las ventanas."
        )
    )

    run = st.button(
        "Ejecutar predicciones",
        type="primary",
        use_container_width=True
    )


# ============================================================
# INFORMACION
# ============================================================

st.info(
    "Esta demostración realiza inferencia con modelos "
    "previamente entrenados. No realiza entrenamiento "
    "ni modifica los parámetros del modelo."
)


# ============================================================
# EJECUCION DE INFERENCIA
# ============================================================

if run:

    try:

        with st.spinner(
            "Cargando datos y ejecutando inferencia..."
        ):

            # Cargar dataset
            if source == "Archivos locales":

                if not Path(x_path).is_file():
                    raise FileNotFoundError(
                        f"No existe X_test: {x_path}"
                    )

                x_df = read_csv(
                    str(Path(x_path).resolve())
                )

                y_df = (
                    read_csv(
                        str(Path(y_path).resolve())
                    )
                    if Path(y_path).is_file()
                    else None
                )

            else:

                if x_upload is None:
                    raise ValueError(
                        "Debes subir X_test.csv."
                    )

                x_df = csv_from_upload(x_upload)

                y_df = (
                    csv_from_upload(y_upload)
                    if y_upload
                    else None
                )

            # Metadatos de la GRU (tipo de ataque, filas comunes)
            meta_df = None

            if meta_upload is not None:
                meta_df = csv_from_upload(meta_upload)

            elif meta_path and Path(meta_path).is_file():
                meta_df = read_csv(
                    str(Path(meta_path).resolve())
                )

            if (
                meta_df is not None
                and len(meta_df) != len(x_df)
            ):
                raise ValueError(
                    "X_test y meta_test tienen tamaños "
                    f"diferentes: {len(x_df)} y {len(meta_df)}."
                )

            # Cargar configuracion
            if gru_config is None:
                raise ValueError(
                    "No se pudo cargar gru_config.json."
                )

            x = x_df

            # Cargar etiquetas
            y = (
                extract_labels(y_df)
                if y_df is not None
                else None
            )

            if y is not None and len(y) != len(x):
                raise ValueError(
                    "X_test y y_test tienen tamaños diferentes: "
                    f"{len(x)} y {len(y)}."
                )

            # Inferencia
            gru_dir = str(Path(model_path).resolve())

            # Carga del modelo fuera del tiempo medido
            load_gru_model(
                gru_dir,
                gru_config["archivo"]
            )

            started = time.perf_counter()

            ventanas, row_indices = prepare_gru(
                x_df,
                gru_dir,
                gru_config,
                meta_df,
                solo_comunes
            )

            preds, scores = predict_gru(
                ventanas,
                gru_dir,
                gru_config
            )

            elapsed = time.perf_counter() - started

            y_eval = (
                y[row_indices]
                if y is not None
                else None
            )

            # Resultados individuales
            results = pd.DataFrame({
                "fila_original": row_indices,
                "prediccion": preds,
                "clase_predicha": np.where(
                    preds == 1,
                    "Ataque",
                    "Benigno"
                ),
                "score": scores,
            })

            if y_eval is not None:

                results["etiqueta_real"] = y_eval

                results["correcto"] = (
                    preds == y_eval
                )

            results.insert(
                1,
                "bloque",
                x_df["bloque"].to_numpy()[row_indices]
            )

            if meta_df is not None:

                results.insert(
                    1,
                    "Timestamp",
                    meta_df["Timestamp"].to_numpy()[row_indices]
                )

                results["tipo"] = (
                    meta_df["Label"].to_numpy()[row_indices]
                )

            gru_state = {
                "config": gru_config,
                "dir": gru_dir,
                "solo_comunes": bool(
                    solo_comunes and meta_df is not None
                ),
                "x_df": x_df,
                "meta": meta_df,
            }

            # Guardar resultados en sesion
            st.session_state["demo_results"] = {
                "model": "GRU",
                "predictions": results,
                "labels": y_eval,
                "scores": scores,
                "elapsed": elapsed,
                "n_input": len(x),
                "threshold": threshold,
                "gru": gru_state,
            }

    except Exception as exc:

        st.error(
            f"No se pudo ejecutar la inferencia: {exc}"
        )

        st.info(
            "Revisa el modelo, las dependencias, "
            "las características y el preprocesamiento."
        )


# ============================================================
# VISUALIZACION DE RESULTADOS
# ============================================================

if "demo_results" in st.session_state:

    data = st.session_state["demo_results"]

    result = data["predictions"]
    y_true = data["labels"]

    pred = result["prediccion"].to_numpy()
    scores = data["scores"]

    st.subheader(
        f"Resultados · {data['model']}"
    )

    st.caption(
        f"{len(result):,} flujos evaluados "
        f"de {data['n_input']:,} entradas · "
        f"Inferencia: {data['elapsed']:.3f} s · "
        f"Umbral: {data['threshold']:.4f}"
    )

    # ========================================================
    # METRICAS
    # ========================================================

    if y_true is not None:

        metrics = evaluate(
            y_true,
            pred,
            scores
        )

        cols = st.columns(6)

        for col, (name, value) in zip(
            cols,
            metrics.items()
        ):

            col.metric(
                name,
                (
                    f"{value:.2%}"
                    if np.isfinite(value)
                    else "N/D"
                )
            )

        st.divider()

        # ====================================================
        # MATRIZ Y DISTRIBUCION
        # ====================================================

        left, right = st.columns(2)

        with left:

            fig, cm = draw_confusion(
                y_true,
                pred
            )

            st.pyplot(fig)
            plt.close(fig)

            tn, fp, fn, tp = cm.ravel()

            st.caption(
                f"TP: {tp:,} · "
                f"TN: {tn:,} · "
                f"FP: {fp:,} · "
                f"FN: {fn:,}"
            )

        with right:

            st.markdown(
                "**Distribución de clases**"
            )

            distribution = pd.DataFrame(
                {
                    "Real": np.bincount(
                        y_true,
                        minlength=2
                    ),
                    "Predicho": np.bincount(
                        pred,
                        minlength=2
                    ),
                },
                index=["Benigno", "Ataque"]
            )

            st.bar_chart(
                distribution,
                x_label="Clase",
                y_label="Cantidad de flujos",
                stack=False
            )

            st.markdown(
                "**Errores de clasificación**"
            )

            errors = pd.DataFrame(
                {
                    "Cantidad": [fp, fn]
                },
                index=[
                    "Falsos positivos",
                    "Falsos negativos"
                ]
            )

            st.bar_chart(errors)

        st.divider()

        # ====================================================
        # CURVAS ROC Y PRECISION-RECALL
        # ====================================================

        if len(np.unique(y_true)) == 2:

            c1, c2 = st.columns(2)

            with c1:

                fpr, tpr, _ = roc_curve(
                    y_true,
                    scores
                )

                fig, ax = plt.subplots(
                    figsize=(5.5, 3.5)
                )

                ax.plot(
                    fpr,
                    tpr,
                    label=(
                        f"AUC = "
                        f"{metrics['ROC-AUC']:.4f}"
                    )
                )

                ax.plot(
                    [0, 1],
                    [0, 1],
                    "--",
                    color="gray"
                )

                ax.set(
                    xlabel="False Positive Rate",
                    ylabel="True Positive Rate",
                    title="Curva ROC"
                )

                ax.legend()

                st.pyplot(fig)
                plt.close(fig)

            with c2:

                precision, recall, _ = (
                    precision_recall_curve(
                        y_true,
                        scores
                    )
                )

                fig, ax = plt.subplots(
                    figsize=(5.5, 3.5)
                )

                ax.plot(
                    recall,
                    precision,
                    label=(
                        f"AP = "
                        f"{metrics['AP']:.4f}"
                    )
                )

                ax.set(
                    xlabel="Recall",
                    ylabel="Precision",
                    title="Curva Precision-Recall",
                    xlim=(0, 1),
                    ylim=(0, 1.05)
                )

                ax.legend()

                st.pyplot(fig)
                plt.close(fig)

        else:

            st.warning(
                "Solo existe una clase en y_test. "
                "No se pueden calcular ROC-AUC ni AP."
            )

        if data.get("gru") is not None:

            st.divider()

            show_gru_section(
                data,
                y_true,
                pred,
                scores
            )

    # ========================================================
    # SIN ETIQUETAS
    # ========================================================

    else:

        st.warning(
            "No se proporcionó y_test.csv. "
            "Se muestran predicciones sin métricas."
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Benignos predichos",
            f"{(pred == 0).sum():,}"
        )

        c2.metric(
            "Ataques predichos",
            f"{(pred == 1).sum():,}"
        )

        st.bar_chart(
            pd.Series(
                np.bincount(
                    pred,
                    minlength=2
                ),
                index=["Benigno", "Ataque"]
            )
        )

    # ========================================================
    # PREDICCIONES INDIVIDUALES
    # ========================================================

    st.divider()

    st.subheader(
        "Predicciones individuales"
    )

    st.dataframe(
        result.head(1000),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Descargar todas las predicciones (CSV)",
        result.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            f"predicciones_"
            f"{data['model'].lower()}.csv"
        ),
        mime="text/csv"
    )


# ============================================================
# INSTRUCCIONES INICIALES
# ============================================================

else:

    st.subheader("Instrucciones")

    st.markdown(
        """
        1. Coloca `X_test.csv`, `y_test.csv` y `meta_test.csv`
        en `Data_Test/gru_test/`.

        2. El umbral y las características se recuperan
           automáticamente del archivo
           `Models/GRU/gru_config.json` (umbral F2 de validación,
           ventana y características); `X_test.csv` trae los
           flujos sin escalar y la demo aplica el log con signo,
           el escalador de entrenamiento y las ventanas por bloque.

        3. Presiona **Ejecutar predicciones**.

        4. Visualiza las métricas, la matriz de
           confusión y las gráficas.

        **Importante:** el modelo debe utilizar
        exactamente el preprocesamiento y la
        configuración de su entrenamiento.
        """
    )
