# ADR 0019: Protocolo de evaluación en test

- **Estado:** Aceptado
- **Fecha:** 2026-10-11
- **Reemplaza a:** la parte de evaluación en prueba del ADR 0005 (archivado)

## Contexto

El conjunto de prueba solo sirve como estimación del desempeño si no interviene en ninguna decisión del modelo. Además, las variantes `csv` y `aleatorio` de las secuencias descartan filas distintas al inicio de cada bloque (ADR 0012), así que sus filas objetivo no coinciden del todo. Para comparar el GRU con SVM y XGBoost, los tres deben evaluarse sobre las mismas filas.

## Decisión

La evaluación en prueba se hace en `notebooks/gru/03_evaluacion_test.ipynb`, con estas reglas:

- **Sin decisiones sobre prueba.** No se entrena ni se ajusta nada con estos resultados. El modelo principal ya estaba designado (ADR 0017) y la configuración, fijada (ADR 0018).
- **Modelos.** Los 15 ya entrenados: 5 semillas de GRU W=10 (orden aleatorio), GRU W=10 (orden CSV) y GRU W=1.
- **Umbral.** El `umbral_f2` elegido en validación para cada semilla (ADR 0015), sin reelegirlo.
- **Filas comunes.** Los tres modelos se evalúan sobre la intersección de las filas objetivo de las variantes `csv` y `aleatorio`. Los índices se guardan en `results/gru/03_evaluacion_test/idx_test_comun.npy`.
- **Comprobaciones antes de predecir.** Etiquetas iguales en las filas comunes, misma fila objetivo en las tres entradas y coincidencia por features entre las filas del CSV y el último paso de cada secuencia.
- **Métricas.** ROC-AUC y PR-AUC; y, en el umbral, F2, recall, precision, F1, accuracy y la matriz de confusión (`metricas_con_umbral()`). También el recall por tipo de ataque. Se reportan por semilla y como media ± desviación estándar (ddof = 1).
- **Artefactos.** `test_resultados.json`, las tablas en CSV, las probabilidades de cada modelo y semilla (`{modelo}_seed{seed}_test_proba.npy`) y las figuras, todo en `results/gru/03_evaluacion_test/`.

La evaluación en prueba se ejecutó una vez. Después el notebook se reejecutó dos veces, con los mismos modelos, los mismos umbrales y predicciones deterministas:

1. para agregar F1, accuracy, TP y TN a `metricas_con_umbral()`;
2. para generar las figuras.

Ninguna de las dos reejecuciones cambió los modelos, los umbrales ni las filas evaluadas. No hubo cambios de diseño después de ver los resultados de prueba.

### Filas evaluadas

| Dato | Valor | Fuente |
|---|---:|---|
| Secuencias de prueba por variante | 66,499 | Output de la sección 2 de `03_evaluacion_test.ipynb` |
| Filas comunes evaluadas | 66,454 | `test_resultados.json`, campo `n_filas` |
| Ataques | 1,164 | `test_resultados.json`, campo `n_ataques` |
| DoS slowloris / DoS Slowhttptest | 914 / 250 | `test_resultados.json`, campo `ataques_por_tipo` |
| Benignos | 65,290 | Output de la sección 2 de `03_evaluacion_test.ipynb` |
| Ataques fuera de la intersección, `csv` / `aleatorio` | 2 / 9 | Mismo output |

## Alternativas consideradas

- **Evaluar cada modelo sobre todas sus secuencias.** Descartada: las variantes tendrían conjuntos de filas distintos y las cifras no serían comparables.
- **Reelegir el umbral en prueba.** Descartada: contaminaría la evaluación (ADR 0015).
- **Evaluar solo el modelo principal.** Descartada: las referencias son necesarias para interpretar su resultado (ADR 0017).
- **Corregir casos atípicos después de ver prueba**, como el umbral de 0.999 de una semilla de GRU W=1. Descartada por la misma razón que reelegir el umbral.

## Consecuencias

- **SVM y XGBoost deben evaluarse sobre las mismas filas.** `idx_test_comun.npy` contiene el índice de `X_test` que devuelve `train_test_split_by_flow()` con sus valores por defecto, así que basta `X_test.loc[idx_test_comun]`.
- La evaluación deja fuera 45 de las 66,499 secuencias de cada variante, entre ellas 2 ataques de la variante `csv` y 9 de la `aleatorio`.
- Todo análisis posterior sobre prueba es descriptivo y no puede motivar cambios en el modelo (ADR 0020).
- Las probabilidades guardadas permiten recalcular métricas y figuras sin volver a predecir.
