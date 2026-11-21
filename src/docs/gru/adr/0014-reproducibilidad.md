# ADR 0014: Reproducibilidad: semillas y determinismo

- **Estado:** Aceptado
- **Fecha:** 2026-10-11

## Contexto

El resultado de entrenar una red depende de la inicialización de pesos, del Dropout y del orden de los batches. Con una sola corrida no se puede saber si una diferencia entre dos configuraciones se debe a la configuración o al azar. En Keras 3, además, los inicializadores y el Dropout toman su semilla del generador `random` de Python en el momento de construir el modelo, así que fijar solo NumPy y TensorFlow no basta.

## Decisión

- **Cinco semillas por experimento:** 0, 1, 2, 3 y 4. Todo resultado se reporta como media ± desviación estándar muestral (ddof = 1) de las cinco.
- **`fijar_semillas(seed)`** (`src/models/gru/train.py`) llama a `keras.utils.set_random_seed(seed)` y a `tf.config.experimental.enable_op_determinism()`. Se llama **antes** de construir el modelo.
- **Una sola entrada para entrenar:** `correr_experimento()` (`src/models/gru/experiments.py`). Por semilla ejecuta, en este orden: `keras.backend.clear_session()`, `fijar_semillas()`, construcción, entrenamiento, guardado y métricas de validación.
- **Artefactos por semilla:** el modelo (`{nombre}_seed{seed}.keras`), las curvas de entrenamiento, las probabilidades de validación y un `{nombre}_seeds.json` con la configuración, las semillas, la fila de cada semilla, la media y la desviación.
- **Datos deterministas:** la partición no usa azar (ADR 0010) y la permutación de la variante `aleatorio` usa la semilla fija 42 (ADR 0012).

## Alternativas consideradas

- **Una sola semilla.** Descartada: no permite estimar la variación entre corridas.
- **Fijar las semillas dentro de la función de entrenamiento.** Descartada: para entonces el modelo ya está construido y sus pesos iniciales ya se sortearon.
- **Más de cinco semillas.** No se adoptó por costo de cómputo; cinco bastan para reportar una desviación estándar.

## Consecuencias

- Los 15 modelos (3 variantes × 5 semillas) están en `models/gru/01_baseline/` y `models/gru/02_ablaciones/` y se versionan.
- La desviación entre semillas es la referencia para juzgar diferencias entre variantes (ADR 0016).
- El determinismo del entrenamiento fue verificado manualmente en CPU (Windows) antes del entrenamiento; no se guardó el output de esa prueba.
- El determinismo de operaciones puede hacer más lento el entrenamiento.
- `requirements.txt` solo fija versiones mínimas. Reproducir los mismos números exige el mismo entorno; el repositorio no guarda las versiones exactas.
- `04_analisis_errores.ipynb` comprueba que las probabilidades guardadas reproducen los falsos negativos y falsos positivos de cada semilla reportados en `test_resultados.json` (output de su sección 1).
