# ADR 0016: Ablaciones y reglas de decisión fijadas antes de correr

- **Estado:** Aceptado
- **Fecha:** 2026-10-11
- **Reemplaza a:** la comparación de ventanas del ADR 0008 (archivado)

## Contexto

El GRU W=10 con el orden del archivo (variante `csv`, ADR 0012) obtiene en validación un F2 de 0.9969 ± 0.0014 (`results/gru/01_baseline/baseline_w10_seeds.json`). Antes de atribuir ese resultado a que el modelo aprovecha la secuencia de flujos, hay que responder dos preguntas:

1. ¿Aporta la ventana de flujos anteriores, o basta el flujo objetivo?
2. ¿Depende el resultado del orden de las filas dentro de cada minuto, que viene del archivo y no del tiempo?

## Decisión

Se entrenan dos ablaciones con la misma configuración (ADR 0013) y las mismas semillas (ADR 0014) que el baseline, y se comparan **solo en validación** (`notebooks/gru/02_ablaciones.ipynb`):

- **W=1:** el último paso de cada ventana `csv`. Mismas filas objetivo y mismas etiquetas, sin los 9 flujos anteriores.
- **Orden aleatorio:** variante `aleatorio` de W=10. Mismas filas en cada partición, con el orden sorteado dentro de cada minuto.

**Reglas de decisión, fijadas antes de entrenar.** Se compara la media de cada ablación con la media y la desviación estándar del baseline:

- "La secuencia aporta" si la media de `recall_05` o la de `pr_auc` de W=1 es menor que media_base − 2·std_base.
- "Hay dependencia del orden del CSV" con el mismo criterio, aplicado al orden aleatorio.
- Si ninguna se cumple: "sin diferencia detectable en validación".

Las reglas se registraron en el commit `1ca4056` (2026-10-08), anterior a los commits con los resultados de las ablaciones (`a35d402` y `15b988c`, 2026-10-09).

### Resultado (`results/gru/02_ablaciones/comparacion.json`)

| Modelo | `recall_05` | `pr_auc` | F2 | FN en umbral F2 | FP en umbral F2 |
|---|---:|---:|---:|---:|---:|
| GRU W=10 (orden CSV) | 0.9959 ± 0.0012 | 0.9995 ± 0.0003 | 0.9969 ± 0.0014 | 6.8 ± 3.3 | 2.4 ± 1.3 |
| GRU W=1 | 0.9805 ± 0.0161 | 0.9931 ± 0.0013 | 0.9809 ± 0.0105 | 14.6 ± 25.9 | 124.0 ± 109.1 |
| GRU W=10 (orden aleatorio) | 0.9759 ± 0.0057 | 0.9966 ± 0.0014 | 0.9789 ± 0.0054 | 20.8 ± 10.6 | 117.2 ± 17.9 |

Los umbrales de la regla son 0.9934 en `recall_05` y 0.9990 en `pr_auc` (output de la sección 4 de `02_ablaciones.ipynb`). Las dos ablaciones quedan por debajo en ambas métricas, así que los veredictos son:

- **W=1: "la secuencia aporta".**
- **Orden aleatorio: "hay dependencia del orden del CSV".**

### Qué representa el orden del archivo

La sección 5 de `02_ablaciones.ipynb` revisa el orden del archivo:

- No está en orden cronológico: en el CSV limpio (443,614 filas, en el orden del archivo), el 30.6 % de las filas consecutivas retrocede más de 120 s, tanto por inicio como por fin del flujo.
- En la captura filtrada por clase (451,326 filas; el CSV limpio no conserva IPs), dentro de un mismo minuto, dos filas consecutivas comparten `Source IP` el 68.6 % de las veces (21.7 % esperado al azar) y `Destination IP` el 47.1 % (15.4 % esperado).

El archivo agrupa los flujos por host. En la variante `csv`, los flujos previos de cada ventana suelen venir del mismo host que el flujo objetivo.

## Alternativas consideradas

- **Decidir a la vista de los resultados, sin regla previa.** Descartada: permite acomodar la interpretación al resultado.
- **Comparar en prueba.** Descartada: prueba se reserva para una única evaluación (ADR 0019).
- **Una regla basada en F2.** Descartada: F2 depende del umbral elegido por semilla; `recall_05` y `pr_auc` no.
- **Reordenar los flujos por su orden real dentro del minuto.** No es posible: el `Timestamp` tiene resolución de un minuto.

## Consecuencias

- En validación, la ventaja del GRU W=10 (orden CSV) sobre W=1 se explica por el agrupamiento por host del archivo y no por un patrón temporal entre flujos. Ese orden no está disponible fuera del archivo.
- Con el orden sorteado, W=10 queda al nivel de W=1: F2 de 0.979 frente a 0.981 y PR-AUC de 0.9966 frente a 0.9931. Esta comparación no tenía regla fijada y es solo descriptiva.
- Estos resultados motivan la elección del modelo principal (ADR 0017) y la decisión de no buscar hiperparámetros (ADR 0018).
- Los nuevos experimentos deben fijar su regla de decisión antes de entrenar y juzgarse en validación.
