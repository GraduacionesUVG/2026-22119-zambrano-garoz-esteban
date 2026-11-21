"""
Métricas, barrido de umbrales de decisión y figuras de resultados del GRU, a
partir de probabilidades ya calculadas.

Funciones puras y reutilizables desde notebook. La selección del umbral en
validación está en docs/gru/adr/0015-seleccion-de-umbral-en-validacion.md y la
evaluación en test, en docs/gru/adr/0019-protocolo-de-evaluacion-en-test.md.

Uso típico:
    from src.models.gru.evaluation import metricas_con_umbral, metricas_desde_probabilidades

    # Validación: elige el umbral de F2 máximo y reporta las métricas en él
    metricas_val = metricas_desde_probabilidades(y_val, proba_val)

    # Test: métricas con el umbral ya elegido en validación
    metricas_test = metricas_con_umbral(y_test, proba_test, metricas_val["umbral_f2"])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

BETA_F2 = 2

# Estilo de las figuras de resultados: una serie por color en orden fijo, rampa de un
# solo tono para magnitudes y texto siempre en tinta, nunca en el color de la serie
DPI_FIGURAS = 200
_COLORES_SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
_RAMPA_SECUENCIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
_TINTA = "#0b0b0b"
_TINTA_SECUNDARIA = "#52514e"
_REJILLA = "#e4e3df"

_CAMPOS_FILA = ["threshold", "precision", "recall", "f1", "f2", "tn", "fp", "fn", "tp"]


def sweep_thresholds_desde_probabilidades(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    """
    Barre una grid fina de umbrales de decisión sobre probabilidades ya
    calculadas y obtiene, para cada umbral, las métricas que usa
    select_threshold_strategies().

    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas binarias reales.
    y_proba : np.ndarray
        Probabilidad predicha de la clase ataque, 1D, en el mismo orden.

    Returns
    -------
    pd.DataFrame
        Una fila por umbral (grid np.arange(0.001, 1.000, 0.001)), columnas:
        threshold, precision, recall, f1, f2, tn, fp, fn, tp. f2 es F-beta con
        beta=2 (pondera recall el doble que precision).
    """
    thresholds = np.arange(0.001, 1.000, 0.001)

    filas = []
    for umbral in thresholds:
        y_pred = (y_proba >= umbral).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        denominador_f2 = (BETA_F2 ** 2) * precision + recall
        f2 = (
            (1 + BETA_F2 ** 2) * precision * recall / denominador_f2
            if denominador_f2 > 0
            else 0.0
        )

        filas.append({
            "threshold": float(umbral),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "f2": float(f2),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

    return pd.DataFrame(filas)


def _fila_a_dict(fila: pd.Series) -> dict:
    resultado = {}
    for campo in _CAMPOS_FILA:
        valor = fila[campo]
        resultado[campo] = int(valor) if campo in ("tn", "fp", "fn", "tp") else float(valor)
    return resultado


def _estrategia_no_alcanzable(piso_recall: float) -> dict:
    resultado = {campo: None for campo in _CAMPOS_FILA}
    resultado["nota"] = (
        f"Ningún umbral de la grid evaluada alcanza recall >= {piso_recall} "
        "sobre este conjunto de validación."
    )
    return resultado


def select_threshold_strategies(df: pd.DataFrame) -> dict:
    """
    Reduce el barrido de sweep_thresholds_desde_probabilidades() a 8 estrategias
    de selección de umbral, cada una respondiendo una pregunta distinta sobre qué
    priorizar.

    Parameters
    ----------
    df : pd.DataFrame
        Salida de sweep_thresholds_desde_probabilidades().

    Returns
    -------
    dict[str, dict]
        8 claves, cada una con threshold/precision/recall/f1/f2/tn/fp/fn/tp
        tomados de la fila correspondiente de df:

        - "f1_maximo": f1 máximo en toda la grid.
        - "f2_maximo": f2 máximo en toda la grid (pondera recall el doble).
        - "recall_min_0.90"/"0.95"/"0.97": entre los umbrales que alcanzan ese
          piso de recall, el de mayor precision (empate -> threshold más alto).
          Si ningún umbral de la grid alcanza el piso, todos los campos
          numéricos quedan en None y se agrega una clave "nota" explicando por qué.
        - "costo_5x"/"10x"/"20x": el umbral que minimiza (factor * fn + fp),
          es decir, el costo asumiendo que un falso negativo pesa `factor`
          veces más que un falso positivo.
    """
    estrategias: dict[str, dict] = {}

    estrategias["f1_maximo"] = _fila_a_dict(df.loc[df["f1"].idxmax()])
    estrategias["f2_maximo"] = _fila_a_dict(df.loc[df["f2"].idxmax()])

    for piso in (0.90, 0.95, 0.97):
        nombre = f"recall_min_{piso:.2f}"
        candidatos = df[df["recall"] >= piso]
        if candidatos.empty:
            estrategias[nombre] = _estrategia_no_alcanzable(piso)
        else:
            # Mayor precision primero; empate en precision -> threshold más alto.
            mejor = candidatos.sort_values(
                ["precision", "threshold"], ascending=[False, False]
            ).iloc[0]
            estrategias[nombre] = _fila_a_dict(mejor)

    for factor in (5, 10, 20):
        nombre = f"costo_{factor}x"
        costo = factor * df["fn"] + df["fp"]
        estrategias[nombre] = _fila_a_dict(df.loc[costo.idxmin()])

    return estrategias


def metricas_desde_probabilidades(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Métricas de un modelo a partir de sus probabilidades: independientes del
    umbral (roc_auc, pr_auc), con umbral 0.5 (recall_05, precision_05) y en el
    umbral de F2 máximo de select_threshold_strategies() (umbral_f2, f2,
    recall_f2, precision_f2, fn_f2, fp_f2).

    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas binarias reales.
    y_proba : np.ndarray
        Probabilidad predicha de la clase ataque, 1D, en el mismo orden.
    """
    y_pred_05 = (y_proba >= 0.5).astype(int)
    f2 = select_threshold_strategies(
        sweep_thresholds_desde_probabilidades(y_true, y_proba)
    )["f2_maximo"]

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "recall_05": float(recall_score(y_true, y_pred_05, zero_division=0)),
        "precision_05": float(precision_score(y_true, y_pred_05, zero_division=0)),
        "umbral_f2": f2["threshold"],
        "f2": f2["f2"],
        "recall_f2": f2["recall"],
        "precision_f2": f2["precision"],
        "fn_f2": f2["fn"],
        "fp_f2": f2["fp"],
    }


def metricas_con_umbral(y_true: np.ndarray, y_proba: np.ndarray, umbral: float) -> dict:
    """
    Métricas con un umbral de decisión ya fijado (no lo elige): roc_auc y pr_auc,
    y en ese umbral f2, recall, precision, fn, fp, f1, accuracy, tp y tn. Sirve
    para evaluar en test con el umbral elegido en validación.

    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas binarias reales.
    y_proba : np.ndarray
        Probabilidad predicha de la clase ataque, 1D, en el mismo orden.
    umbral : float
        Se predice ataque si y_proba >= umbral.
    """
    y_pred = (y_proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    denominador_f2 = (BETA_F2 ** 2) * precision + recall
    f2 = (1 + BETA_F2 ** 2) * precision * recall / denominador_f2 if denominador_f2 > 0 else 0.0

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f2": float(f2),
        "recall": float(recall),
        "precision": float(precision),
        "fn": int(fn),
        "fp": int(fp),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tp": int(tp),
        "tn": int(tn),
    }


def _estilo_ejes(ax, rejilla: str = "both") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_REJILLA)
    ax.tick_params(colors=_TINTA_SECUNDARIA, length=0)
    ax.grid(axis=rejilla, color=_REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def _guardar_figura(fig: Figure, save_path: str | None) -> None:
    if save_path:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DPI_FIGURAS, bbox_inches="tight", facecolor="white")


def plot_matrices_confusion(conteos: dict[str, np.ndarray], save_path: str | None = None) -> Figure:
    """
    Una matriz de confusión por modelo con la media ± desviación estándar (ddof=1)
    de cada celda entre semillas. El color es la proporción dentro de la clase real
    (fila), para que la clase minoritaria no quede aplastada por la mayoritaria.

    Parameters
    ----------
    conteos : dict[str, np.ndarray]
        modelo -> array (n_semillas, 2, 2) con [[tn, fp], [fn, tp]] de cada semilla.
    save_path : str, opcional
        Si se pasa, guarda la figura como PNG.
    """
    clases = ["Benigno", "Ataque"]
    cmap = LinearSegmentedColormap.from_list("secuencial", _RAMPA_SECUENCIAL)
    fig, axes = plt.subplots(
        1, len(conteos), figsize=(4.6 * len(conteos), 4.4), sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, (nombre, matrices) in zip(axes, conteos.items()):
        media = matrices.mean(axis=0)
        std = matrices.std(axis=0, ddof=1)
        proporcion = media / media.sum(axis=1, keepdims=True)
        imagen = ax.imshow(proporcion, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{media[i, j]:,.1f} ± {std[i, j]:,.1f}\n{proporcion[i, j]:.2%}",
                    ha="center", va="center", fontsize=10,
                    color="white" if proporcion[i, j] > 0.5 else _TINTA,
                )
        ax.set_title(nombre, color=_TINTA)
        ax.set_xticks([0, 1], clases)
        ax.set_yticks([0, 1], clases)
        ax.set_xticks([0.5], minor=True)
        ax.set_yticks([0.5], minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0, colors=_TINTA_SECUNDARIA)
        ax.spines[:].set_visible(False)
        ax.set_xlabel("Predicción")
    axes[0].set_ylabel("Clase real")

    barra = fig.colorbar(imagen, ax=axes, shrink=0.8, label="Proporción dentro de la clase real")
    barra.outline.set_visible(False)
    fig.suptitle("Matrices de confusión en test (media ± desv. estándar entre semillas)", color=_TINTA)

    _guardar_figura(fig, save_path)
    return fig


def plot_curvas_por_semilla(
    y_true: np.ndarray,
    probas: dict[str, dict[int, np.ndarray]],
    tipo: str = "roc",
    fpr_log: bool = False,
    save_path: str | None = None,
) -> Figure:
    """
    Curvas ROC (tipo="roc") o precision-recall (tipo="pr") en un panel por modelo,
    con una curva por semilla y su área bajo la curva en la leyenda.

    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas binarias reales.
    probas : dict[str, dict[int, np.ndarray]]
        modelo -> {semilla: probabilidades 1D en el mismo orden que y_true}.
    tipo : {"roc", "pr"}
    fpr_log : bool
        Solo para ROC: eje de falsos positivos en escala logarítmica, útil cuando
        las curvas quedan pegadas a la esquina superior izquierda.
    save_path : str, opcional
        Si se pasa, guarda la figura como PNG.
    """
    if tipo not in ("roc", "pr"):
        raise ValueError(f"tipo debe ser 'roc' o 'pr' (recibido {tipo!r})")
    n_semillas = max(len(por_semilla) for por_semilla in probas.values())
    if n_semillas > len(_COLORES_SERIE):
        raise ValueError(f"Hay {n_semillas} semillas y solo {len(_COLORES_SERIE)} colores de serie")

    fig, axes = plt.subplots(
        1, len(probas), figsize=(4.8 * len(probas), 4.6), sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, (nombre, por_semilla) in zip(axes, probas.items()):
        for color, (seed, y_proba) in zip(_COLORES_SERIE, por_semilla.items()):
            if tipo == "roc":
                x, y, _ = roc_curve(y_true, y_proba)
                etiqueta = f"Semilla {seed} (AUC = {roc_auc_score(y_true, y_proba):.5f})"
            else:
                y, x, _ = precision_recall_curve(y_true, y_proba)
                etiqueta = f"Semilla {seed} (PR-AUC = {average_precision_score(y_true, y_proba):.4f})"
            ax.plot(x, y, color=color, linewidth=1.8, label=etiqueta)

        if tipo == "roc":
            if fpr_log:
                ax.set_xscale("log")
                ax.set_xlim(1 / int((np.asarray(y_true) == 0).sum()), 1)
                ax.set_xlabel("Tasa de falsos positivos (escala logarítmica)")
            else:
                ax.plot([0, 1], [0, 1], color=_TINTA_SECUNDARIA, linewidth=1, linestyle="--", label="Azar")
                ax.set_xlabel("Tasa de falsos positivos")
        else:
            ax.set_xlabel("Recall")
        ax.set_title(nombre, color=_TINTA)
        ax.legend(
            loc="lower right" if tipo == "roc" else "lower left",
            frameon=False, fontsize=9, labelcolor=_TINTA_SECUNDARIA,
        )
        _estilo_ejes(ax)

    axes[0].set_ylabel("Tasa de verdaderos positivos (recall)" if tipo == "roc" else "Precisión")
    fig.suptitle(
        "Curvas ROC en test, por semilla" if tipo == "roc"
        else "Curvas precisión-recall en test, por semilla",
        color=_TINTA,
    )

    _guardar_figura(fig, save_path)
    return fig


def plot_recall_por_tipo(
    media: pd.DataFrame,
    minimo: pd.DataFrame,
    maximo: pd.DataFrame,
    save_path: str | None = None,
) -> Figure:
    """
    Recall por tipo de ataque: dentro de cada modelo (índice), una barra por tipo
    (columnas) con la media entre semillas y un bigote del mínimo al máximo.

    Parameters
    ----------
    media, minimo, maximo : pd.DataFrame
        Mismo índice (modelo) y mismas columnas (tipo de ataque).
    save_path : str, opcional
        Si se pasa, guarda la figura como PNG.
    """
    tipos = list(media.columns)
    if len(tipos) > len(_COLORES_SERIE):
        raise ValueError(f"Hay {len(tipos)} tipos y solo {len(_COLORES_SERIE)} colores de serie")
    ancho = 0.8 / len(tipos)
    posiciones = np.arange(len(media))

    fig, ax = plt.subplots(figsize=(1.6 + 2.2 * len(media), 4.4), constrained_layout=True)
    for k, (tipo, color) in enumerate(zip(tipos, _COLORES_SERIE)):
        x = posiciones + (k - (len(tipos) - 1) / 2) * ancho
        ax.bar(
            x, media[tipo], width=ancho, color=color, edgecolor="white", linewidth=2, label=tipo,
            yerr=[media[tipo] - minimo[tipo], maximo[tipo] - media[tipo]],
            capsize=4, error_kw={"ecolor": _TINTA, "elinewidth": 1.2},
        )
        for xi, m, tope in zip(x, media[tipo], maximo[tipo]):
            ax.text(xi, tope + 0.015, f"{m:.3f}", ha="center", va="bottom", fontsize=9, color=_TINTA)

    ax.set_xticks(posiciones, [str(modelo).replace(" (", "\n(") for modelo in media.index])
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Recall en test")
    ax.set_title("Recall por tipo de ataque (media; bigotes: mínimo y máximo entre semillas)", color=_TINTA)
    ax.legend(
        title="Tipo de ataque", frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5),
        labelcolor=_TINTA_SECUNDARIA,
    )
    _estilo_ejes(ax, rejilla="y")

    _guardar_figura(fig, save_path)
    return fig


def plot_barras_por_minuto(
    tabla: pd.DataFrame,
    valor: str,
    grupo: str,
    titulo: str,
    etiqueta_y: str,
    anotacion: str | None = None,
    save_path: str | None = None,
) -> Figure:
    """
    Una barra por minuto, coloreada según un grupo (por ejemplo, el tipo de ataque).

    Parameters
    ----------
    tabla : pd.DataFrame
        Una fila por minuto, con el minuto (Timestamp) como índice y ya ordenada.
    valor : str
        Columna con la altura de cada barra.
    grupo : str
        Columna categórica que define el color y la leyenda.
    titulo, etiqueta_y : str
    anotacion : str, opcional
        Columna con el texto que se escribe sobre cada barra.
    save_path : str, opcional
        Si se pasa, guarda la figura como PNG.
    """
    grupos = list(dict.fromkeys(tabla[grupo]))
    if len(grupos) > len(_COLORES_SERIE):
        raise ValueError(f"Hay {len(grupos)} grupos y solo {len(_COLORES_SERIE)} colores de serie")
    color_de = dict(zip(grupos, _COLORES_SERIE))
    posiciones = np.arange(len(tabla))

    fig, ax = plt.subplots(figsize=(max(6.0, 2.0 + 0.55 * len(tabla)), 4.4), constrained_layout=True)
    for g in grupos:
        es_grupo = (tabla[grupo] == g).to_numpy()
        ax.bar(
            posiciones[es_grupo], tabla.loc[es_grupo, valor], width=0.7, color=color_de[g],
            edgecolor="white", linewidth=2, label=g,
        )
    tope = float(tabla[valor].max())
    if anotacion is not None:
        for x, y, texto in zip(posiciones, tabla[valor], tabla[anotacion]):
            ax.text(x, y + 0.02 * tope, str(texto), ha="center", va="bottom", fontsize=9, color=_TINTA)

    ax.set_xticks(posiciones, [f"{minuto:%H:%M}" for minuto in tabla.index], rotation=45, ha="right")
    ax.set_ylim(0, 1.15 * tope if tope > 0 else 1)
    ax.set_xlabel("Minuto de inicio del flujo")
    ax.set_ylabel(etiqueta_y)
    ax.set_title(titulo, color=_TINTA)
    ax.legend(title=grupo.capitalize(), frameon=False, labelcolor=_TINTA_SECUNDARIA)
    _estilo_ejes(ax, rejilla="y")

    _guardar_figura(fig, save_path)
    return fig
