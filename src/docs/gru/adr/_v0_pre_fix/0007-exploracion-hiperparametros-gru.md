**Archivado:** corresponde al pipeline anterior a la corrección del preprocesamiento compartido. Ver ADR-0018.

# ADR 0007: Exploración de hiperparámetros del GRU (W=10)

## Estado

Aceptado.

## Contexto

El protocolo (Fase 6) pide evaluar configuraciones de arquitectura y ajustar hiperparámetros. Se corrió una exploración mínima de 2 variantes sobre W=10 y se compararon con el baseline (`docs/gru/adr/0005-ajuste-umbral-decision-gru.md`) solo en validación. El conjunto de test no se tocó.

La regla de decisión se fijó antes de ver resultados (`notebooks/gru/gru_hyperparameter_search.ipynb`). Una variante reemplaza al baseline solo si cumple dos condiciones a la vez: mejora el F2 máximo en val en al menos 0.001 y baja la `val_loss` respecto al baseline.

## Resultados (validación)

| Configuración | val_loss | F2 máx | Umbral F2 | F1 máx | Parámetros | Épocas |
|---|---:|---:|---:|---:|---:|---|
| baseline_32 (GRU 32, dropout 0.3) | 0.00135 | 0.9967 | 0.05 | 0.9946 | 9,345 | 27 (restaurada 22) |
| gru_64 | 0.00320 | 0.9922 | 0.438 | 0.9863 | 24,833 | 7 (restaurada 2) |
| gru_32_16 (2 capas apiladas) | 0.00894 | 0.9665 | 0.323 | 0.9722 | 11,729 | 9 (restaurada 4) |

Además, `gru_32_16` se volvió inestable en las épocas tardías, con `val_auc` cayendo a ~0.96.

## Decisión

Se conserva `baseline_32`. Ninguna variante cumple la regla, porque las dos empeoran tanto el F2 máximo como la `val_loss`.

## Limitaciones

- Se usó un solo seed por configuración.
- Las corridas no son comparables en épocas. El baseline corrió 27 (restaurada 22), mientras que EarlyStopping cortó las variantes en 7 y 9 épocas, guiado por una `val_loss` todavía ruidosa en esas primeras épocas.

## Consecuencias

- No cambian el modelo ni el resultado de test de `docs/gru/adr/0005-ajuste-umbral-decision-gru.md`.
- La comparación W=10 vs. W=20 se hará con la arquitectura del baseline.
- No se evaluó ninguna variante en test, para mantener test reservado a una única evaluación final (ver `docs/gru/adr/0005-ajuste-umbral-decision-gru.md`).