# ADR 0012: Construcción de secuencias e índices objetivo

- **Estado:** Aceptado
- **Fecha:** 2026-10-11
- **Reemplaza a:** `mecanismo-mapeo-secuencias.md` (archivado)

## Contexto

El GRU recibe ventanas de flujos consecutivos. Hay que definir cómo se forman, qué etiqueta lleva cada una y cómo se relaciona cada ventana con una fila del dataset limpio. Esa relación es necesaria para evaluar los tres modelos del proyecto sobre las mismas filas y para analizar errores con las features originales.

El `Timestamp` tiene resolución de un minuto y cada minuto contiene cientos de flujos. El orden dentro del minuto solo puede venir del archivo.

## Decisión

`scripts/gru/build_gru_sequences.py` genera todas las secuencias a partir del CSV procesado, con `build_sequences()` de `src/models/gru/sequences.py`:

- **Ventana.** W = 10 flujos consecutivos, con paso 1. La etiqueta de la ventana es la de su último flujo, que es la **fila objetivo**.
- **Fronteras.** Una ventana solo usa filas de la misma partición y del mismo bloque (ADR 0010). Las primeras W − 1 filas de cada bloque no generan ventana.
- **Índice objetivo.** Junto a cada `X_{split}_w10.npy` e `y_{split}_w10.npy` se guarda `idx_{split}_w10.npy`, con el índice de la fila objetivo de cada ventana en el mismo orden. Es el índice de `X` que devuelve `train_test_split_by_flow()`: la posición de la fila en el dataset limpio ordenado de forma estable por `Timestamp`.
- **Variantes.** Se generan dos, con las mismas filas en cada partición y el mismo escalador (ADR 0011):
  - `csv` (archivos sin sufijo): conserva el orden del archivo dentro de cada minuto.
  - `aleatorio` (sufijo `_aleatorio`): sortea el orden de las filas dentro de cada (partición, bloque, `Timestamp`), con semilla 42. Ninguna fila cambia de partición: en los minutos que cruzan una frontera (10:03, 10:07, 10:29 y 10:33), qué filas quedan en cada partición lo sigue decidiendo el orden del archivo.
- **Verificación antes de guardar.** Cada trío X/y/idx se valida antes de escribirlo: shape `(n, 10, 59)`, sin NaN ni infinitos, número de ataques igual al esperado y etiquetas de las filas de `idx` iguales a `y`. La variante `aleatorio` se compara fila a fila con `csv`.

La entrada W = 1 de la ablación (ADR 0016) no se genera aparte: es el último paso de cada ventana `csv`.

### Cifras

| Dato | Valor | Fuente |
|---|---|---|
| Secuencias de entrenamiento | (310484, 10, 59) | Output de la sección 2 de `01_baseline.ipynb` |
| Secuencias de validación | (66496, 10, 59) | Mismo output |
| Secuencias de prueba (cada variante) | 66,499 | Output de la sección 2 de `03_evaluacion_test.ipynb` |
| Ataques en entrenamiento, `csv` / `aleatorio` | 8,230 / 8,224 | Outputs de las secciones 2 y 3 de `02_ablaciones.ipynb` |
| Ataques en validación, ambas variantes | 1,884 | Mismos outputs |

## Alternativas consideradas

- **Reconstruir la fila objetivo repitiendo el recorrido de `build_sequences()`.** Descartada: exige volver a ejecutar la partición y depende de que nada haya cambiado. Guardar el índice es más simple y se puede verificar.
- **Guardar el número de fila del CSV en lugar del índice de la partición.** Descartada: SVM y XGBoost reciben sus datos de `train_test_split_by_flow()`, y con ese índice pueden seleccionar las mismas filas con `X_test.loc[idx]`.
- **Agrupar las ventanas por host o por conexión.** No es posible: el dataset limpio no conserva IPs ni puertos (ADR 0009).
- **Generar también W = 20.** Descartada: ningún experimento vigente la usa (ADR 0018).

## Consecuencias

- El script escribe 18 archivos `.npy` y un escalador.
- Las dos variantes descartan filas distintas al inicio de cada bloque, así que sus filas objetivo no coinciden del todo. La evaluación en prueba usa la intersección (ADR 0019).
- El orden dentro de cada ventana `csv` es el del archivo, que no es cronológico dentro del minuto (ADR 0016).
- `data/sequences/` no se versiona; se regenera con el script.
