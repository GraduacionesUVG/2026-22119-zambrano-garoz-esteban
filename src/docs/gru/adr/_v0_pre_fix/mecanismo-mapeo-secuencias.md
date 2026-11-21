**Archivado:** corresponde al pipeline anterior a la corrección del preprocesamiento compartido. Ver ADR-0012.

# Mecanismo de mapeo secuencia -> fila original

`src/models/gru/error_analysis.py` usa este mecanismo para traducir posiciones de secuencias (por ejemplo, índices en `X_test_w10.npy`) a su fila original en el CSV/dataframe.

Los `.npy` de `data/sequences/` (por ejemplo `X_test_w10.npy`) solo contienen las 63 columnas de `FEATURE_COLUMNS` (`src/models/gru/sequences.py`). `train_test_split_by_flow()` conserva `Timestamp` y `bloque` como metadata, pero esas columnas se descartan al construir las secuencias. Junto a los `.npy` tampoco se guarda ninguna columna de índice. Por eso la secuencia en la posición `i` de `X_test_w10.npy` no tiene ninguna referencia directa a su fila original.

Para recuperarla, el módulo no intenta deducir la aritmética de posiciones. Reproduce exactamente el recorrido de `build_sequences()` (`src/models/gru/sequences.py`): usa el mismo orden de bloques con `sorted(bloque.unique())` y el mismo `range(window_size - 1, n)` dentro de cada bloque. La diferencia es que, en lugar de acumular arrays de features, anota qué fila original (índice de pandas, bloque, Timestamp) corresponde a la posición fin de cada ventana. Esa es justo la fila que `build_sequences()` usa como etiqueta de la secuencia. Su docstring lo dice: "la etiqueta es y en esa misma posición".

`construir_metadata_secuencias()` devuelve una tabla con una fila por secuencia, en el mismo orden que `X_seq/y_seq` de `build_sequences()`. La fila `i` de esa tabla corresponde 1:1 a la secuencia `i` de `X_test_w10.npy/y_test_w10.npy`. Para que eso se cumpla, hay que pasarle el mismo `X_test`, con `Timestamp` y `bloque` intactos, y el mismo `window_size` que se usaron para generar las secuencias.

Ese `X_test` con `Timestamp` y `bloque` intactos, el que `build_sequences()` consumió originalmente, se reconstruye llamando otra vez a `load_and_clean()` y `train_test_split_by_flow()` sobre `data/raw`. Las dos funciones son deterministas: no usan aleatoriedad y el split es posicional por `Timestamp` (ver `docs/gru/adr/0001-split-cronologico-por-bloques-temporales.md`). Por lo tanto reproducen el mismo `X_test` que generó los `.npy` actuales, siempre que `data/raw` no haya cambiado desde entonces. Antes de confiar en el mapeo conviene comparar `len(X_test)` con el shape real del `.npy` (ver `mapear_a_metadata()`).

La columna `Label` original, que distingue Slowloris de Slowhttptest, no sobrevive a `train_test_split_by_flow()`. Esa función la excluye explícitamente de X (`columnas_a_excluir` incluye `"Label"`), así que se recupera por separado.

Primero, sobre el dataframe de `load_and_clean()` y antes de llamar a `train_test_split_by_flow()`, se aplica el mismo ordenamiento cronológico que esa función hace internamente: `sort_values('Timestamp', kind='stable')` y luego `reset_index(drop=True)`. Después se une por índice de pandas. `train_test_split_by_flow()` conserva ese índice sin resetearlo en `X_train`, `X_val` y `X_test`, así que los dos dataframes quedan alineados.