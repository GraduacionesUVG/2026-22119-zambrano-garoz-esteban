# ADR 0020: Análisis de errores post hoc

- **Estado:** Aceptado
- **Fecha:** 2026-10-11
- **Reemplaza a:** ADR 0006 (archivado)

## Contexto

Las métricas agregadas de prueba no dicen qué flujos falla el modelo ni si los errores se repiten entre semillas. Esa información ayuda a interpretar los resultados y a describir las limitaciones del trabajo. Como se obtiene de los datos de prueba, no puede usarse para cambiar el modelo.

## Decisión

El análisis de errores se hace en `notebooks/gru/04_analisis_errores.ipynb`, con estas reglas:

- **Post hoc y descriptivo.** No entrena, no predice y no cambia umbrales ni resultados. No se toma ninguna decisión del modelo con este análisis.
- **Entradas.** Las probabilidades de prueba ya guardadas, `idx_test_comun.npy`, los `umbral_f2` de validación y el CSV procesado, del que se toman las features originales, el `Timestamp` y el tipo de ataque.
- **Modelos.** GRU W=10 (orden aleatorio) como principal y GRU W=1 como contraste.
- **Error persistente.** El que ocurre en al menos 3 de las 5 semillas. Cada fila se clasifica como TP, FN, FP o TN por la predicción mayoritaria.
- **Comprobaciones.** Las etiquetas del CSV coinciden con las de las secuencias, y los falsos negativos y falsos positivos por semilla coinciden con `test_resultados.json`.
- **Contenido.** Errores persistentes por modelo; falsos negativos por tipo de ataque, minuto y posición en el tramo de ataque; falsos positivos por minuto; perfil de features por tipo de resultado; coincidencia de errores entre los dos modelos.

### Hallazgos (`results/gru/04_analisis_errores/`)

| Hallazgo | Cifras | Archivo |
|---|---|---|
| Errores persistentes del modelo principal | 27 FN (17 en las 5 semillas) y 115 FP (49 en las 5) | `errores_persistentes.csv` |
| Errores persistentes de GRU W=1 | 8 FN y 50 FP; de 212 ataques fallados en alguna semilla, 203 lo son en una sola | `errores_persistentes.csv` |
| FN persistentes por tipo de ataque | 21 de Slowhttptest (8.4 % de 250) y 6 de Slowloris (0.7 % de 914) | `fn_por_tipo.csv` |
| FN persistentes por posición | Slowloris: los 6 en el tramo final. Slowhttptest: 3 al inicio, 2 en el medio y 16 al final | `fn_por_posicion.csv` |
| FP persistentes por tipo de minuto | 77 en los 10 minutos con ataques (0.92 % de sus benignos) y 38 en los 82 minutos sin ataques (0.067 %) | `fp_por_tipo_de_minuto.csv` |
| Duración mediana del flujo | TP 8.5 s, FN 192 µs, FP 1.7 s, TN 31 ms | `perfil_features.csv` |
| Coincidencia con GRU W=1 | FN: 8 en ambos, 19 solo en el principal, 0 solo en W=1. FP: 32, 83 y 18 | `coincidencia_errores.csv` |

## Alternativas consideradas

- **Analizar los errores de validación.** No se adoptó: el interés está en describir los errores del resultado final, que es el de prueba.
- **Analizar cada semilla por separado.** Descartada como criterio principal: mezcla los errores del modelo con los que dependen de la semilla. La persistencia los separa.
- **Usar los hallazgos para ajustar el modelo o el umbral.** Descartada: sería decidir con datos de prueba (ADR 0019).

## Consecuencias

- Los hallazgos se reportan como descripción y como limitaciones, no como evidencia para rediseñar el modelo.
- Los falsos negativos persistentes son flujos cortos que por sus features se parecen al tráfico benigno, y los falsos positivos persistentes se parecen a los ataques detectados. Esto es coherente con la decisión del ADR 0018, pero se obtuvo después de ella.
- El dataset procesado no incluye IPs, así que no se puede comprobar si los falsos positivos van dirigidos al servidor atacado.
- Los patrones por minuto son descriptivos; el análisis no los pone a prueba.
