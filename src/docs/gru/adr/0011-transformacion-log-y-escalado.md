# ADR 0011: Transformación logarítmica con signo y escalado

- **Estado:** Aceptado
- **Fecha:** 2026-10-11
- **Reemplaza a:** la parte de escalado del ADR 0001 (archivado)

## Contexto

Las features de duración, tiempos entre paquetes y tasas abarcan varios órdenes de magnitud. Con estandarización sola, casi todos los valores quedan comprimidos cerca de cero y unos pocos valores extremos dominan la escala. El GRU es sensible a la escala de sus entradas.

Algunas columnas tienen valores negativos válidos (por ejemplo, −1 en `Init_Win_bytes_backward`), para los que `log1p` no está definido.

## Decisión

`scripts/gru/build_gru_sequences.py` llama a `scale_features(..., transformacion_log=True)`:

1. Aplica `sign(x)·log1p(|x|)` a las 59 features.
2. Ajusta un `StandardScaler` solo con entrenamiento y aplica esa misma transformación a validación y prueba.

El escalador ajustado se guarda en `data/sequences/scaler_csv.joblib`. Hay un único escalador: las dos variantes de secuencias (ADR 0012) usan las mismas filas ya escaladas.

La constante `TRANSFORMACION_LOG = True` del script fija el mismo valor que usa el pipeline del SVM, para que ambos modelos reciban la misma transformación.

## Alternativas consideradas

- **Solo `StandardScaler`.** Descartada por la compresión descrita en el contexto.
- **`log1p` sin signo.** Descartada: produce −∞ en x = −1.
- **Ajustar el escalador con todo el dataset.** Descartada: filtraría información de validación y prueba al entrenamiento.
- **Un escalador por variante de secuencias.** Descartada: las variantes tienen las mismas filas en cada partición, así que un segundo escalador sería idéntico.

## Consecuencias

- Para aplicar un modelo a datos nuevos hace falta el mismo paso de logaritmo y `scaler_csv.joblib`.
- `notebooks/gru/03_evaluacion_test.ipynb` (sección 3) reproduce la transformación sobre las filas del CSV y comprueba que coincide con el último paso de cada secuencia de prueba. Su output confirma la coincidencia en las 66,454 filas evaluadas.
- Si el SVM cambia su transformación, este valor debe revisarse para mantener la comparación.
