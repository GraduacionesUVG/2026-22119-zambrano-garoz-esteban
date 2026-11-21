# ADR 0017: Modelo principal y modelos de referencia

- **Estado:** Aceptado
- **Fecha:** 2026-10-11

## Contexto

Hay tres variantes del GRU entrenadas con la misma configuración: W=10 con el orden del archivo, W=10 con orden aleatorio dentro de cada minuto y W=1. Hay que decidir cuál representa al GRU en la tesis y en la comparación con SVM y XGBoost. Las ablaciones mostraron que el resultado de W=10 con el orden del archivo depende de ese orden, que agrupa los flujos por host y no es cronológico (ADR 0016).

## Decisión

- **Modelo principal: GRU W=10 (orden aleatorio)**, `aleatorio_w10`. Es el modelo con ventana cuyo resultado no depende del orden del archivo.
- **GRU W=10 (orden CSV)**, `baseline_w10`: referencia. Es una **cota optimista**, porque aprovecha el agrupamiento por host del archivo, que no estaría disponible al aplicar el modelo fuera de él.
- **GRU W=1**, `w1`: referencia sin ventana. Mide lo que se obtiene solo con las features del flujo objetivo.

El modelo principal se designó **antes de evaluar en prueba**. Consta en la introducción de `notebooks/gru/03_evaluacion_test.ipynb` (commit `9c04d49`, 2026-10-10) y en el campo `modelo_principal` de `results/gru/03_evaluacion_test/test_resultados.json`.

**El modelo principal no se cambia a partir de resultados de prueba.**

## Alternativas consideradas

- **GRU W=10 (orden CSV) como principal.** Descartada: su desempeño depende de un orden que es un artefacto del archivo.
- **GRU W=1 como principal.** Descartada: no usa secuencia, y el objeto del trabajo es un modelo recurrente sobre ventanas de flujos. Se conserva como referencia para medir cuánto aporta la ventana.
- **Elegir el principal según el mejor resultado en prueba.** Descartada: sería seleccionar el modelo con los datos de prueba.

## Consecuencias

- Las cifras del GRU que se comparan con SVM y XGBoost son las de GRU W=10 (orden aleatorio).
- En prueba, el modelo principal no es el de mejores métricas (`results/gru/03_evaluacion_test/tabla_test_resumen.csv`):

  | Modelo | Recall | Precision | F1 | PR-AUC |
  |---|---:|---:|---:|---:|
  | GRU W=10 (orden aleatorio) | 0.973 ± 0.011 | 0.897 ± 0.022 | 0.933 ± 0.011 | 0.985 ± 0.002 |
  | GRU W=10 (orden CSV) | 0.987 ± 0.021 | 0.995 ± 0.004 | 0.991 ± 0.013 | 0.999 ± 0.001 |
  | GRU W=1 | 0.958 ± 0.078 | 0.949 ± 0.035 | 0.951 ± 0.034 | 0.991 ± 0.002 |

  Esto no modifica la designación.
- La tesis debe reportar los tres modelos con su rol, y explicar por qué el de mejores cifras es una cota optimista.
- Con orden aleatorio, la ventana de 10 flujos no supera al flujo individual. Es un resultado del trabajo y una limitación de la configuración evaluada (ADR 0018).
