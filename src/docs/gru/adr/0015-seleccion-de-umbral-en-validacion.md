# ADR 0015: Selección del umbral de decisión en validación

- **Estado:** Aceptado
- **Fecha:** 2026-10-11
- **Reemplaza a:** ADR 0005 (archivado)

## Contexto

El GRU produce una probabilidad de ataque y hace falta un umbral para convertirla en decisión. El proyecto prioriza el recall de la clase ataque, porque un ataque no detectado cuesta más que una falsa alarma. El umbral es un ajuste del modelo, así que no puede elegirse con los datos de prueba.

## Decisión

- El umbral se elige **en validación y por semilla**.
- Se barre una grid de 0.001 a 0.999 en pasos de 0.001 y se toma el umbral que maximiza **F2** (F-beta con beta = 2, que pondera el recall el doble que la precision). Lo implementan `sweep_thresholds_desde_probabilidades()` y `select_threshold_strategies()["f2_maximo"]`, a través de `metricas_desde_probabilidades()`, en `src/models/gru/evaluation.py`.
- El umbral queda en el campo `umbral_f2` de cada fila de `por_semilla` en los `*_seeds.json`.
- En prueba se aplica ese mismo umbral, sin reelegirlo (ADR 0019), con `metricas_con_umbral()`.
- Además se reportan métricas que no dependen del umbral (ROC-AUC y PR-AUC) y el recall y la precision con umbral 0.5.

### Umbrales elegidos (`umbral_f2` en los `*_seeds.json`)

| Modelo | Semilla 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| GRU W=10 (orden aleatorio) | 0.106 | 0.125 | 0.066 | 0.070 | 0.103 |
| GRU W=10 (orden CSV) | 0.076 | 0.239 | 0.835 | 0.146 | 0.744 |
| GRU W=1 | 0.010 | 0.472 | 0.113 | 0.507 | 0.999 |

## Alternativas consideradas

- **Umbral fijo de 0.5.** Descartada como criterio principal: con 0.5, la precision de validación de dos semillas del GRU W=10 (orden CSV) baja a 0.958 y 0.978 (conclusiones de `01_baseline.ipynb`). Se conserva como métrica de referencia.
- **Maximizar F1.** Descartada: pondera igual recall y precision, y el proyecto prioriza el recall.
- **Un único umbral para las cinco semillas.** Descartada: cada modelo tiene su propia distribución de probabilidades.
- **Elegir el umbral en prueba.** Descartada: contaminaría la evaluación.

## Consecuencias

- El umbral varía mucho entre semillas en GRU W=10 (orden CSV) y GRU W=1, y poco en GRU W=10 (orden aleatorio).
- Un umbral extremo elegido en validación puede no trasladarse a prueba. En la semilla 4 de GRU W=1, el umbral de 0.999 deja 212 falsos negativos en prueba (`results/gru/03_evaluacion_test/tabla_test_por_semilla.csv`). No se corrige, porque hacerlo sería elegir el umbral con datos de prueba.
- Las métricas independientes del umbral permiten comparar modelos sin este efecto.
- Si se reentrena un modelo, su umbral debe volver a elegirse en validación.
