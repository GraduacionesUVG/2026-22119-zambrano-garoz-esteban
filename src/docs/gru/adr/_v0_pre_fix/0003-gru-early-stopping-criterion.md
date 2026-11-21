**Archivado:** corresponde al pipeline anterior a la corrección del preprocesamiento compartido. Ver ADR-0013.

# ADR 0003: Criterio de monitoreo para EarlyStopping del GRU baseline

## Estado

Aceptado.

## Contexto

El EarlyStopping de `train_gru_model()` (`src/models/gru/train.py`) monitoreaba originalmente `val_recall` (`mode='max'`, `patience=5`). Con solo 2,109 ejemplos de ataque en validación (de 67,513 filas totales), el recall a umbral fijo (0.5) demostró ser una señal ruidosa que oscila fuertemente entre épocas: un puñado de predicciones cerca del umbral puede mover el recall varios puntos porcentuales de una época a la siguiente, sin que eso refleje una mejora o deterioro real del modelo.

## Corrida 1 — criterio `val_recall` (descartada como criterio de parada, documentada aquí como evidencia)

El entrenamiento se detuvo en la época 6, con un patrón de `val_recall` claramente oscilante: 1.00 en la época 1, cayendo a ~0.93 en las épocas 2-3, subiendo a ~0.96 en la época 5, y volviendo a bajar. Los pesos restaurados (`restore_best_weights=True`) correspondieron a la época 1 (el punto de mejor `val_recall`), pese a que esa misma época tuvo el peor `val_loss` de todo el entrenamiento (~0.016). Es decir: el criterio eligió como "mejor" el modelo menos entrenado y menos calibrado de la corrida, solo porque coincidió con un pico ruidoso de recall.

Métricas de test de esa corrida:

| Métrica   |  Valor |
| --------- | -----: |
| Accuracy  | 0.9932 |
| Precision | 0.7065 |
| Recall    | 0.9973 |
| F1-score  | 0.8271 |
| ROC-AUC   | 0.9998 |

Matriz de confusión `[[tn, fp], [fn, tp]]`: `[[65952, 457], [3, 1100]]`: 3 falsos negativos, 457 falsos positivos.

## Decisión

Cambiar el monitor de EarlyStopping a `val_loss` (`mode='min'`), manteniendo `patience=5` y `restore_best_weights=True`. `val_loss` se calcula sobre probabilidades continuas y las 67,513 filas completas de validación (no solo los ejemplos de la clase positiva), lo que la hace una señal mucho más estable para decidir cuándo detener el entrenamiento.

Esto es un rol distinto al de métrica de decisión del proyecto: recall en la clase ataque sigue siendo la métrica principal para comparar modelos entre sí (falsos negativos son el error costoso, ver `src/evaluation/metrics.py`) y para la comparación de Fase 7-8. Este cambio solo afecta el criterio de parada temprana durante el entrenamiento del GRU, no qué métrica importa para evaluar el modelo ya entrenado.

## Corrida 3 — criterio `val_loss` (baseline oficial)

El entrenamiento corrió 27 épocas, deteniéndose por falta de mejora en `val_loss`, con pesos restaurados de la época 22 (mejor `val_loss`: 0.0014).

Métricas de test de esta corrida:

| Métrica   |  Valor |
| --------- | -----: |
| Accuracy  | 0.9972 |
| Precision | 0.9740 |
| Recall    | 0.8486 |
| F1-score  | 0.9070 |
| ROC-AUC   | 0.9996 |

Matriz de confusión `[[tn, fp], [fn, tp]]`: `[[66384, 25], [167, 936]]`: 167 falsos negativos, 25 falsos positivos.

## Trade-off observado

El criterio `val_loss` produce un modelo con mejor generalización general (precision, F1, muchas menos falsas alarmas: 25 FP vs. 457) pero con recall crudo notablemente menor (0.8486 vs. 0.9973): 167 ataques no detectados en vez de 3.

Dado que el protocolo del proyecto prioriza explícitamente el recall de la clase ataque como métrica principal (con F1 como chequeo secundario contra falsas alarmas, ver `src/evaluation/metrics.py`), este trade-off es relevante y no se descarta sin más. Se aborda mediante ajuste del umbral de decisión post-entrenamiento (no reentrenando), en `notebooks/gru/gru_feature_engineering.ipynb`, usando el modelo ya entrenado (`models/gru/baseline_w10.keras`). El ROC-AUC de 0.9996 de la corrida 3 indica que el modelo ordena bien las probabilidades incluso con este criterio de parada, por lo que mover el umbral es la vía más directa para recuperar recall sin necesidad de reentrenar ni volver al criterio `val_recall`.

## Decisión final

Se conserva la corrida 3 (`val_loss`) como el baseline oficial, congelado en `notebooks/gru/gru_model_training.ipynb`, con el entendido de que el ajuste de umbral en el siguiente notebook es el paso que resuelve la prioridad de recall del protocolo, sin reintroducir la inestabilidad de monitorear `val_recall` durante el entrenamiento.

## Alternativas consideradas

* Mantener `val_recall` como criterio de parada. Descartada: la corrida 1 demuestra que selecciona pesos en función de ruido (época 1, con el peor `val_loss` de la corrida), no de una mejora real del modelo.
* Monitorear `val_recall` con un umbral distinto a 0.5 para estabilizar la señal. Descartada por ahora: cambia dos variables a la vez (umbral y criterio de parada) sin aislar cuál resuelve el problema; el ajuste de umbral ya está planeado como paso posterior y separado, sobre el modelo ya entrenado con `val_loss`.
* Aumentar `patience` para que `val_recall` tenga más margen de absorber el ruido. Descartada: no ataca la causa raíz (pocos positivos en validación → señal ruidosa a umbral fijo), solo retrasa cuándo se materializa el mismo problema.
* Reentrenar con un umbral de decisión distinto desde el inicio en vez de ajustar post-entrenamiento. Descartada: el ROC-AUC de 0.9996 ya confirma que el modelo de la corrida 3 ordena bien las probabilidades, así que reentrenar no aporta nada que el ajuste de umbral no resuelva más barato.

## Consecuencias

* `train_gru_model()` (`src/models/gru/train.py`) usa `val_loss`/`mode='min'` como monitor de EarlyStopping de forma permanente para el GRU baseline; `val_recall` no se usa como criterio de parada en ningún notebook de este modelo.
* El baseline oficial (corrida 3, `models/gru/baseline_w10.keras`) tiene recall de test (0.8486) por debajo de lo que el protocolo consideraría ideal como cifra final. Esto es una limitación conocida y documentada del baseline, no un resultado definitivo. Se aborda en `notebooks/gru/gru_feature_engineering.ipynb` vía ajuste de umbral.
* Ambas corridas quedan documentadas en `results/gru/baseline_w10_early_stopping_comparison.json` para trazabilidad, aunque solo la corrida 3 es la que se usa a partir de aquí.