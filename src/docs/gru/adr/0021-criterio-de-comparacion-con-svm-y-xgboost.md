# ADR 0021: Criterio de comparación con SVM y XGBoost

- **Estado:** Aceptado
- **Fecha:** 2026-10-15

## Contexto

La comparación con SVM y XGBoost necesita una regla para decir, en cada indicador, si el GRU es superior, equivalente o inferior a cada modelo tradicional. Los tres modelos se evaluaron sobre las mismas 66,454 filas de prueba (`idx_test_comun.npy`, ADR 0019). El GRU tiene cinco semillas (ADR 0014); SVM y XGBoost tienen una sola corrida cada uno.

SVM y XGBoost también eligieron su umbral en validación maximizando F2, restringido a umbrales con recall de al menos 0.75; XGBoost toma además la mediana sobre 1,000 remuestreos bootstrap de validación. Sus resultados en las filas comunes están en ramas que no forman parte de `feature/gru`:

| Modelo | Rama | Archivo |
|---|---|---|
| SVM | `SVM` | `reports/svm/metricas_test_comun_completa.json` (umbral: `criterio_umbral` de `reports/svm/resultados_completa.json`) |
| XGBoost | `feature/xgboost-modelo` | `reports/xgboost/resultados_test.json`, campo `subconjunto_comun_gru` (umbral: `src/models/xgboost/model.py`) |

**La regla no se fijó antes de evaluar.** Se definió al consolidar los resultados, después de conocer las métricas de prueba de SVM y XGBoost.

## Decisión

Para cada indicador se compara el valor de prueba de SVM y de XGBoost con el intervalo media ± 2·std (ddof = 1) de las cinco semillas del modelo principal, GRU W=10 (orden aleatorio) (ADR 0017):

- Dentro del intervalo: **equivalente**.
- Fuera del intervalo: el GRU es **superior** o **inferior** según el sentido del indicador.

| Indicador | Mejor si es |
|---|---|
| Recall, precision, F1-score, F2, accuracy, ROC-AUC, PR-AUC | mayor |
| Falsos negativos, falsos positivos, tasa de falsos negativos | menor |

Las métricas que dependen del umbral se toman en el umbral que cada modelo eligió en validación.

El veredicto del tercer objetivo específico se basa en recall, F1-score y ROC-AUC, las métricas que define ese objetivo. Los demás indicadores se clasifican con la misma regla y se reportan como complementarios.

El tiempo de entrenamiento no se clasifica: el del GRU es por semilla y el de SVM corresponde a un único ajuste final, sin la búsqueda de hiperparámetros.

La regla tiene la misma forma que la de las ablaciones (ADR 0016), con tres diferencias:

- se aplica en ambos sentidos, no solo hacia abajo;
- la referencia es el GRU y el valor comparado es una sola corrida, no la media de cinco semillas;
- no se registró antes de evaluar.

## Alternativas consideradas

- **Comparar los valores puntuales sin tolerancia.** Descartada: ignora la variación entre semillas del GRU.
- **Un criterio distinto del de las ablaciones.** No se adoptó, para no introducir un criterio nuevo después de ver los resultados.
- **Fijar la regla antes de evaluar.** Ya no era posible cuando se definió.

## Consecuencias

- Las clasificaciones son descriptivas. La regla se eligió conociendo los resultados, así que no tiene el valor de una regla registrada de antemano.
- El intervalo solo refleja la variación del GRU; la de SVM y XGBoost entre corridas es desconocida.
- La tabla de la sección 8 de `docs/gru/resultados_gru.md` debe aplicar esta regla.
