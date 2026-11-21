**Archivado:** corresponde al pipeline anterior a la corrección del preprocesamiento compartido. Ver ADR-0013.

# ADR 0004: Estrategia de balanceo de clases distinta por modelo

## Estado

Aceptado.

## Contexto

El dataset CIC-IDS-2017 presenta un desbalance severo entre tráfico benigno y tráfico de ataque (~97-98% benigno / 2-3% ataque, según el split; ver `docs/gru/adr/0001-split-cronologico-por-bloques-temporales.md` para los porcentajes exactos por split). La tabla de riesgos del protocolo de investigación menciona genéricamente "técnicas de balanceo si es necesario" como mitigación, sin especificar SMOTE ni ninguna técnica en particular. La elección concreta queda a criterio del equipo.

El equipo trabaja con tres modelos de naturaleza distinta:

* GRU (secuencial), a cargo de Esteban Zambrano (`src/models/gru/`)
* SVM (tabular), a cargo de Juan Pablo Solís (`src/models/svm/`)
* XGBoost (tabular), a cargo de Edwin Ortega (`src/models/xgboost/`)

## Decisión

Cada modelo usa la estrategia de balanceo más adecuada a su propia naturaleza, no una técnica única impuesta a los tres:

| Modelo  | Estrategia                                                                | Motivo                                                                                                                                                                                                      |
| ------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SVM     | SMOTE (oversampling sintético) sobre `X_train` tabular                    | Técnica estándar para datos tabulares; cada fila es una observación independiente, condición que SMOTE asume.                                                                                               |
| XGBoost | `scale_pos_weight` (parámetro nativo de boosting)                         | Práctica estándar recomendada para XGBoost en problemas desbalanceados; no requiere generar datos sintéticos.                                                                                               |
| GRU     | `class_weight` en la función de pérdida (`binary_crossentropy`), no SMOTE | Ver justificación técnica abajo. Con los datos actuales de train (307,139 clase 0 / 8,083 clase 1), da aproximadamente `{0: 0.513, 1: 19.50}` (ver `compute_class_weights()` en `src/models/gru/train.py`). |

## Justificación técnica de excluir SMOTE para el GRU

SMOTE genera ejemplos sintéticos interpolando linealmente entre dos filas tabulares reales de la clase minoritaria. Esto asume que cada observación es independiente entre sí.

Las secuencias de entrada al GRU no son filas independientes: son ventanas temporales de forma `(10, 63)`, 10 pasos consecutivos en el tiempo, construidas por `build_sequences()` (`src/models/gru/sequences.py`). Aplicar SMOTE directamente sobre estas secuencias interpolaría entre dos ventanas temporales completas de ataques distintos, produciendo una secuencia sintética que mezcla partes de dos eventos de ataque sin coherencia temporal real. El resultado no representaría un patrón de tráfico plausible, sino un artefacto sin sentido secuencial, el tipo de dato que el modelo nunca vería en producción.

`class_weight`, en cambio, no genera datos sintéticos: ajusta el costo de cada error durante el entrenamiento, forzando al modelo a prestar más atención a los ejemplos reales (no inventados) de la clase minoritaria.

## Evidencia de que el volumen de datos reales no es una limitación

El GRU baseline tiene solo 9,345 parámetros entrenables (arquitectura pequeña: `GRU(32) + Dropout + Dense`, ver `src/models/gru/model.py`). Con 8,083/2,109/1,103 secuencias de ataque reales en train/val/test respectivamente, el modelo alcanzó ROC-AUC de 0.9998 (corrida con criterio `val_recall`) y 0.9996 (corrida con criterio `val_loss`, ver `docs/gru/adr/0003-gru-early-stopping-criterion.md`). Esta es evidencia empírica de que el modelo distingue casi perfectamente ambas clases con los datos disponibles, sin necesidad de aumentar el volumen de datos mediante generación sintética.

## Limitación reconocida de `class_weight` (por sí solo)

`class_weight` mejora el aprendizaje durante el entrenamiento pero no controla el umbral de decisión final. La cobertura completa de la prioridad de recall que exige el protocolo depende de combinar `class_weight` con el ajuste de umbral post-entrenamiento (ver `notebooks/gru/gru_feature_engineering.ipynb` y `docs/gru/adr/0003-gru-early-stopping-criterion.md`).

## Alternativas consideradas

* SMOTE sobre las secuencias del GRU, igual que en SVM. Descartada: produciría secuencias sintéticas sin coherencia temporal (ver justificación técnica arriba): un artefacto, no un patrón de tráfico plausible.
* Aplicar la misma técnica de balanceo (SMOTE) a los tres modelos por consistencia metodológica superficial. Descartada: la consistencia que importa para la comparación de Fase 7-8 es la del split y las métricas de evaluación (ver ADR 0001), no la técnica de balanceo interna de cada modelo. Forzar SMOTE sobre datos secuenciales sería metodológicamente peor, no mejor.
* Técnicas de data augmentation específicas para series temporales (jittering, time warping), que perturban una secuencia real en vez de interpolar entre dos secuencias distintas. Se documentan como posible trabajo futuro, no como necesidad actual: los resultados ya obtenidos (ROC-AUC ~0.999) no muestran limitación por volumen de datos que las justifique en este momento.

## Consecuencias

* El equipo usa tres estrategias de balanceo distintas, justificadas por la naturaleza de cada modelo (secuencial vs. tabular), no por inconsistencia metodológica. Esto debe mencionarse explícitamente en la discusión metodológica de Fase 9 para anticipar la pregunta de por qué no se usó el mismo enfoque en los tres modelos.
* El recall de test del GRU baseline (0.8486 en la corrida oficial, ver ADR 0003) depende en parte del ajuste de umbral posterior, no solo de `class_weight`. Esa dependencia queda documentada como parte del diseño, no como un pendiente sin resolver.
* Si en el futuro se evalúa una técnica de augmentation para el GRU (jittering, time warping), debería registrarse en un ADR aparte, ya que cambiaría la composición de `X_train`/`y_train` de forma similar en impacto a lo que motivó el ADR 0001.
