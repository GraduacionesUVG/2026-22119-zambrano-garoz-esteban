**Archivado:** corresponde al pipeline anterior a la corrección del preprocesamiento compartido. Ver ADR-0016 y ADR-0018.

# ADR 0008: Comparación de tamaño de ventana W=10 vs. W=20 en el GRU

## Estado

Aceptado.

## Contexto

El protocolo (Fase 5) plantea evaluar tamaños de ventana alternativos. Se entrenó la configuración del baseline (GRU 32, 1 capa, dropout 0.3, class_weight) sobre W=20 y se comparó con W=10 (`docs/gru/adr/0005-ajuste-umbral-decision-gru.md`) solo en validación. El conjunto de test no se tocó.

La regla de decisión se fijó antes de ver resultados (`notebooks/gru/gru_window_comparison.ipynb`). W=20 reemplaza a W=10 solo si cumple dos condiciones a la vez: mejora el F2 máximo en val en al menos 0.001 y baja la `val_loss`.

## Resultados (validación)

| Ventana | val_loss | F2 máx | Umbral F2 | F1 máx | Umbral F1 | Parámetros | Épocas |
|---|---:|---:|---:|---:|---:|---:|---|
| W=10 | 0.001350 | 0.9967 | 0.050 | 0.9946 | 0.337 | 9,345 | 27 (restaurada 22) |
| W=20 | 0.001798 | 0.9944 | 0.373 | 0.9917 | 0.373 | 9,345 | 7 |

## Decisión

Se conserva W=10. W=20 no cumple ninguna de las dos condiciones de la regla.

## Interpretación

El resultado es coherente con `docs/gru/adr/0006-analisis-errores-falsos-negativos-gru.md`. Los falsos negativos del baseline no tienen la señal lenta característica de Low & Slow, así que una ventana más larga no aporta contexto útil para detectarlos.

## Limitaciones

- Se usó un solo seed.
- Las corridas no tienen el mismo número de épocas: W=10 entrenó 27 (restaurada 22) y W=20 solo 7.
- Las secuencias de W=20 tienen 50 filas menos por split que las de W=10, con los mismos ataques (8,083/2,109/1,103).

## Consecuencias

- El modelo y el resultado de test de `docs/gru/adr/0005-ajuste-umbral-decision-gru.md` no cambian.
- No se evaluó W=20 en test.
- Con esta comparación se cierra la Fase 6 del GRU.