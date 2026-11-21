**Archivado:** corresponde al pipeline anterior a la corrección del preprocesamiento compartido. Ver ADR-0020.

# ADR 0006: Análisis de errores de los 137 falsos negativos de test del GRU baseline

## Estado

Aceptado.

## Contexto

El GRU baseline con umbral ajustado (`docs/gru/adr/0005-ajuste-umbral-decision-gru.md`, `threshold=0.050`) no detecta 137 de los 1,103 ataques de test (recall 0.8758). Para ver qué tienen en común esos 137 casos se usaron `src/models/gru/error_analysis.py` y `notebooks/gru/gru_error_analysis.ipynb`. La metadata original de cada caso (`Label`, `bloque`, features crudas) se reconstruyó con el mecanismo determinista que describe el docstring de ese módulo. Primero se reproducen `load_and_clean()` y `train_test_split_by_flow()` sobre `data/raw`. Después, una réplica exacta del recorrido de `build_sequences()` alinea cada secuencia con su fila original.

## Hallazgo 1: distribución por tipo de ataque

| Tipo de ataque | Ataques totales en test | Falsos negativos | Tasa de fallo |
|---|---:|---:|---:|
| DoS slowloris | 914 | 128 | 14.00% |
| DoS Slowhttptest | 189 | 9 | 4.76% |

El modelo falla con Slowloris unas 3 veces más que con Slowhttptest. El 93.43% de los 137 falsos negativos son Slowloris.

## Hallazgo 2: distribución por bloque cronológico

| Bloque | Ataques totales | Falsos negativos | Tasa de fallo |
|---|---:|---:|---|
| Bloque 2 | 913 | 127 | 13.91% |
| Bloque 4 | 189 | 9 | 4.76% |
| Bloque 1 | 1 | 1 | (muestra insignificante, no concluyente) |

Bloque y tipo de ataque están confundidos. Cada tipo de ataque se ejecutó en una ventana horaria específica del dataset original (Sharafaldin et al. 2018, citado en el marco teórico del protocolo). El split cronológico por bloques (`docs/gru/adr/0001-split-cronologico-por-bloques-temporales.md`) asigna el Bloque 2 a la ventana de Slowloris y el Bloque 4 a la de Slowhttptest. Por eso este hallazgo y el Hallazgo 1 no son dos señales independientes. Es la misma señal vista desde dos columnas.

## Hallazgo 3: distribución de probabilidad predicha

| Rango de probabilidad | % de los 137 FN |
|---|---:|
| < 0.001 | 15.33% |
| 0.001 a 0.01 | 69.34% |
| 0.01 a 0.03 | 10.95% |
| 0.03 a 0.05 | 4.38% (los únicos realmente cerca del umbral) |

El modelo no duda con estos 137 casos. Los clasifica como benignos con alta confianza: más del 84% tiene una probabilidad predicha menor que 0.01, muy por debajo del umbral de 0.050. El problema no está en la calibración del umbral. El modelo simplemente no ve estos casos como sospechosos.

## Hallazgo 4: comparación de features crudas (mediana, FN vs. verdaderos positivos)

| Feature | Falsos negativos | Verdaderos positivos | Razón |
|---|---:|---:|---|
| Flow Duration | 9,903 | 4,082,887 | TP dura ~412x más |
| Flow Bytes/s | 48,773.10 | 24.13 | FN transmite ~2,000x más rápido |
| Flow Packets/s | 302.94 | 0.93 | FN envía ~325x más paquetes/segundo |
| Flow IAT Mean | 4,951.5 | 1,501,974 | TP espera ~303x más entre paquetes |

Los verdaderos positivos muestran el patrón low and slow que describe el marco teórico del proyecto: conexiones largas, baja tasa de transmisión y pausas largas entre paquetes. Los 137 falsos negativos no lo muestran y se parecen más a tráfico rápido y de corta duración.

Hay una hipótesis, todavía sin confirmar (ver Próximos pasos). Probablemente son fases de establecimiento de conexión anteriores a que el ataque entre en su comportamiento lento característico. Otra posibilidad es que sean variantes o instancias de la herramienta Slowloris que CICFlowMeter segmentó como flujos cortos independientes.

## Conclusión

El recall de 0.8758 no significa que el modelo no haya aprendido el patrón low-and-slow. Lo aprendió, como muestran los verdaderos positivos del Hallazgo 4. Lo que ocurre es que un subconjunto real de flujos etiquetados como ataque no presenta esa señal temporal. Es una limitación de cómo están representados esos casos en el dataset, no necesariamente del modelo ni de la arquitectura.

Esto también explica por qué el ajuste de umbral (ADR 0005) tuvo poco efecto sobre el recall. Mover el punto de corte solo sirve cuando el modelo está indeciso, con probabilidades intermedias. El Hallazgo 3 muestra que estos 137 casos no son de ese tipo, así que ningún umbral razonable los habría recuperado.

## Próximos pasos sugeridos (no implementados, fuera del alcance de este ADR)

- Explorar si la iteración de hiperparámetros o la comparación `window_size=10` vs. `window_size=20` mejora la detección de estos casos atípicos.
- Inspeccionar a mano una muestra de estos 137 flujos crudos para confirmar la hipótesis de la fase de conexión inicial planteada en el Hallazgo 4.

## Consecuencias

- Este análisis es solo diagnóstico. No cambia el modelo, el umbral ni ningún archivo de `src/models/gru/` aparte de `error_analysis.py`, que ya existía desde una iteración anterior.
- El recall de 0.8758 (ADR 0005) sigue siendo la cifra oficial del baseline. Este ADR explica de dónde sale esa cifra, pero no la reemplaza ni la ajusta.
- Cualquier trabajo futuro sobre estos 137 casos, incluidos los próximos pasos de arriba, debería partir de este ADR para no repetir la investigación desde cero.