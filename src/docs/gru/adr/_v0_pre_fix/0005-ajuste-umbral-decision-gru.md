**Archivado:** corresponde al pipeline anterior a la corrección del preprocesamiento compartido. Ver ADR-0015 y ADR-0019.

# ADR 0005: Ajuste del umbral de decisión del GRU baseline

## Estado

Aceptado.

## Contexto

El GRU baseline (`docs/gru/adr/0003-gru-early-stopping-criterion.md`, corrida oficial con criterio `val_loss`) evaluado con el umbral de decisión por defecto (0.5) obtuvo recall de test de 0.8486 (167 falsos negativos), muy por debajo de la prioridad de recall que exige el protocolo (recall en la clase ataque como métrica principal, ver `src/evaluation/metrics.py`). Sin embargo, su ROC-AUC de test era casi perfecto (0.9996), señal de que el modelo sí ordena bien las probabilidades entre clases. El problema no era la capacidad del modelo, sino dónde se colocaba el punto de corte sobre esas probabilidades.

## Metodología

Barrido de umbrales (`sweep_thresholds()`, `src/models/gru/threshold.py`) de 0.001 a 0.999 en pasos de 0.001, calculado sobre el conjunto de validación (no test), comparando 8 estrategias de selección (`select_threshold_strategies()`):

* F1 máximo.
* F2 máximo (beta=2, pondera recall el doble que precision).
* Recall mínimo garantizado (pisos de 0.90/0.95/0.97), maximizando precisión dentro de cada piso.
* Costo asimétrico: minimizar `5*FN + FP`, `10*FN + FP`, `20*FN + FP`.

Usar validación para elegir el umbral es a propósito: es un ajuste de hiperparámetro, y test debe reservarse para una única evaluación final del umbral ya elegido. Evaluar múltiples umbrales contra test habría invalidado esa evaluación final.

## Resultado del barrido

Las 8 estrategias colapsaron en 3 grupos, sobre validación:

| Grupo | Estrategias                                             | Umbral | Precision | Recall |     F1 |     F2 | FN | FP |
| ----- | ------------------------------------------------------- | -----: | --------: | -----: | -----: | -----: | -: | -: |
| A     | `f1_maximo`                                             |  0.337 |    0.9934 | 0.9957 | 0.9946 | 0.9953 |  9 | 14 |
| B     | `f2_maximo`, `costo_5x`, `costo_10x`, `costo_20x`       |  0.050 |    0.9855 | 0.9995 | 0.9925 | 0.9967 |  1 | 31 |
| C     | `recall_min_0.90`, `recall_min_0.95`, `recall_min_0.97` |  0.938 |    0.9985 | 0.9734 | 0.9858 | 0.9782 | 56 |  3 |

## Hallazgo relevante sobre el Grupo C

Los 3 pisos de recall mínimo dieron el peor recall real de los tres grupos (0.9734, frente a 0.9957 y 0.9995 de A y B). La causa es el diseño del criterio: una vez satisfecho el piso de recall exigido, la regla maximiza precisión, no recall: el nombre de la estrategia ("recall mínimo") no garantiza el resultado que su nombre sugiere si la regla de desempate no lo refleja. Esto queda documentado como una lección de diseño de la estrategia, no como un error de implementación: el criterio hizo exactamente lo que se le pidió (satisfacer el piso y maximizar precisión), solo que eso no coincide con "maximizar recall".

## Decisión

Se elige el Grupo B (umbral = 0.050), alineado con la prioridad explícita del protocolo: recall primero, F1 como chequeo secundario contra falsas alarmas.

Justificación del trade-off frente al Grupo A (F1 máximo, la alternativa más cercana): pasar de A a B gana 8 ataques detectados adicionales en validación (de 9 falsos negativos a 1) a cambio de 17 falsos positivos adicionales sobre 65,404 flujos benignos de validación, una tasa de falsos positivos que sube de 0.021% a 0.047%, un incremento prácticamente insignificante en términos operativos. A cambio, la pérdida en F1 es marginal (0.9946 → 0.9925). Dado que el costo de un falso negativo (un ataque Low & Slow no detectado) es sustancialmente mayor que el de revisar una falsa alarma adicional, este trade-off favorece claramente al Grupo B.

## Resultado final en test (umbral 0.050, evaluado una sola vez)

| Métrica   |  Valor |
| --------- | -----: |
| Accuracy  | 0.9973 |
| Precision | 0.9555 |
| Recall    | 0.8758 |
| F1-score  | 0.9139 |
| ROC-AUC   | 0.9996 |

Matriz de confusión `[[tn, fp], [fn, tp]]`: `[[66364, 45], [137, 966]]` (137 falsos negativos, 45 falsos positivos).

## Hallazgo: brecha de generalización val→test

El umbral 0.050 (Grupo B) no reprodujo en test el desempeño que prometía en validación:

| Conjunto   | Umbral | Recall |  FN | FP |
| ---------- | -----: | -----: | --: | -: |
| Validación |  0.050 | 0.9995 |   1 | 31 |
| Test       |  0.050 | 0.8758 | 137 | 45 |

Esto representa una caída sustancialmente mayor a la esperada para el mismo punto de corte.

El ROC-AUC de test se mantuvo en 0.9996, prácticamente idéntico al de validación. Esto descarta que el modelo se haya degradado o que exista un bug en la evaluación: la capacidad del modelo para ordenar probabilidades entre clases es la misma en ambos conjuntos. El problema es específico del punto de corte elegido, no de la calidad del modelo.

Comparado contra el baseline sin ajustar (umbral 0.5) sobre el mismo test: recall 0.8486 (167 FN) → 0.8758 (137 FN) con el umbral ajustado. El ajuste de umbral sí mejoró el resultado, pero mucho menos de lo que la validación prometía.

Dos causas hipotetizadas:

1. Un umbral extremo (0.050, muy cercano a 0) es inherentemente más sensible a diferencias entre la distribución de probabilidades de validación y de test que un umbral moderado.
2. Val y test provienen de bloques cronológicos distintos, por lo que es esperable cierta variación en las características del tráfico de ataque entre ambos conjuntos.

Hallazgo adicional relevante: incluso con un umbral tan permisivo como 0.050, 137 de 1,103 ataques en test (~12.4%) mantienen una probabilidad predicha por debajo de ese umbral, es decir, el modelo los clasifica con alta confianza como tráfico benigno. Esto sugiere la existencia de un subconjunto de ataques que el modelo no aprendió a reconocer, un problema que el ajuste de umbral no puede resolver por sí solo, ya que no es un problema de calibración del punto de corte sino de qué aprendió el modelo.

Este hallazgo se investigará como siguiente paso, analizando qué características comparten esos 137 casos.

## Alternativas consideradas

* Grupo A (F1 máximo, umbral 0.337). Queda documentado como alternativa igualmente defendible si el contexto priorizara minimizar falsas alarmas sobre maximizar recall, por ejemplo, si cada falsa alarma tuviera un costo operativo real de revisión manual que hiciera indeseable el incremento de FP del Grupo B.
* Grupo C (pisos de recall mínimo 0.90/0.95/0.97). Descartado: pese al nombre, produjo el peor recall de los tres grupos (ver hallazgo arriba). No cumple mejor la prioridad del protocolo que A o B, y su umbral (0.938) es el más alejado de lo que el protocolo busca.
* Mantener el umbral por defecto (0.5). Descartado: es la causa raíz del problema que motiva este ADR (recall 0.8486, 167 FN), pese a que el modelo ya ordena las probabilidades casi perfectamente (ROC-AUC 0.9996).

## Consecuencias

* El umbral 0.050 se aplica únicamente al momento de evaluar, vía `evaluate_gru_model(..., threshold=0.050)` (`src/models/gru/train.py`). No requiere reentrenar el modelo ni modificar `model.py`/`train.py`. El modelo guardado (`models/gru/baseline_w10.keras`) permanece sin cambios.
* Cualquier evaluación futura del GRU baseline W=10 debe usar `threshold=0.050` explícitamente para ser comparable con la cifra reportada en Fase 7-8; el default de `evaluate_gru_model()` (0.5) queda como valor genérico de la función, no como el umbral operativo del proyecto.
* Si se reentrena el modelo (nueva arquitectura, otro `window_size`, etc.), este barrido debe repetirse: el umbral 0.050 es específico de las probabilidades que produce el modelo actual (`baseline_w10.keras`), no una constante del problema.
* Resultados completos del barrido (las 8 estrategias) quedan en `results/gru/threshold_strategies_w10.json` y la gráfica comparativa en `results/gru/threshold_strategies_w10.png`, para trazabilidad.
* El resultado de test (recall 0.8758, no el 0.9995 optimista de validación) es el número oficial y definitivo de esta iteración del baseline. Tras ver esta brecha val→test no se volvió a tocar el umbral ni se buscó uno mejor sobre el propio test: hacerlo habría invalidado la metodología de "evaluar test una sola vez" ya establecida en el proyecto (ver `notebooks/gru/gru_feature_engineering.ipynb`).
