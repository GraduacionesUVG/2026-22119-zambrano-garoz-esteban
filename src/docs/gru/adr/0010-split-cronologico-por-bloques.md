# ADR 0010: Split cronológico por bloques

- **Estado:** Aceptado
- **Fecha:** 2026-10-11
- **Reemplaza a:** ADR 0001 (archivado)

## Contexto

El GRU construye cada entrada con flujos consecutivos, así que necesita particiones de flujos contiguos en el tiempo. Una partición aleatoria repartiría flujos vecinos entre entrenamiento, validación y prueba. Además, los tres modelos del proyecto deben evaluarse sobre la misma partición.

Los ataques ocupan dos ventanas cortas de la mañana. Un corte cronológico único sobre todo el día dejaría algún ataque fuera de alguna partición.

## Decisión

Se usa `train_test_split_by_flow()` de `src/preprocessing/clean.py` con sus valores por defecto:

- El dataset se ordena por `Timestamp` con ordenamiento estable. Como el `Timestamp` tiene resolución de un minuto, dentro de cada minuto se conserva el orden del archivo (`desempate="csv"`).
- Se divide en 5 bloques: antes de Slowloris (1), ventana de Slowloris (2), intervalo entre ataques (3), ventana de Slowhttptest (4) y después de Slowhttptest (5). Las fronteras son el primer y el último `Timestamp` de cada ataque (`outlier_quantile=0.0`).
- Dentro de cada bloque se corta por posición: 70 % para entrenamiento, 15 % para validación y 15 % para prueba.
- No se deja hueco entre particiones (`minutos_hueco=0`).

Antes de generar secuencias, `scripts/gru/build_gru_sequences.py` verifica el split y se detiene sin escribir nada si falla algún criterio:

- los bloques 1, 3 y 5 no tienen ataques;
- cada partición tiene los dos tipos de ataque;
- validación dura al menos 2 minutos en los bloques 2 y 4.

El reporte queda en `results/gru/split_verificacion.txt`.

### Cifras (`results/gru/split_verificacion.txt`)

| Clase | Entrenamiento | Validación | Prueba |
|---|---:|---:|---:|
| BENIGN | 302,299 | 64,657 | 65,369 |
| DoS Slowhttptest | 4,144 | 1,095 | 260 |
| DoS slowloris | 4,086 | 789 | 915 |

| Bloque | Entrenamiento | Validación | Prueba |
|---|---|---|---|
| 2 (Slowloris) | 09:48 – 10:03 | 10:03 – 10:07 | 10:07 – 10:11 |
| 4 (Slowhttptest) | 10:15 – 10:29 | 10:29 – 10:33 | 10:33 – 10:37 |

El reporte termina en "Criterios: OK".

## Alternativas consideradas

- **Partición aleatoria estratificada.** Descartada: reparte flujos vecinos entre las tres particiones, así que las ventanas de cada partición estarían formadas por flujos que no son contiguos en el tiempo.
- **Un solo corte cronológico sobre todo el día.** Descartada: las ventanas de ataque quedarían enteras en una partición.
- **Recortar los extremos de cada ventana con un cuantil mayor que 0.** Descartada: asignaría ataques reales a bloques benignos. Las filas de ataque aisladas ya se excluyen en la limpieza (ADR 0009).
- **Dejar un hueco entre particiones (`minutos_hueco > 0`).** No se adoptó: validación y prueba duran unos 4 minutos por ataque, y un hueco de un minuto quitaría una parte grande de sus ataques. El riesgo que cubriría, fragmentos de una misma conexión a ambos lados de una frontera, queda como limitación.

## Consecuencias

- La proporción de ataques difiere entre particiones. Es efecto del corte por posición y no se corrige.
- Prueba corresponde al último 15 % de cada bloque, es decir, a los últimos minutos de cada ataque. Si el ataque cambia hacia su final, validación no lo refleja.
- El orden cronológico se respeta dentro de cada bloque, no entre bloques: el entrenamiento del bloque 4 (10:15 – 10:29) es posterior a la prueba del bloque 2 (10:07 – 10:11). La evaluación no es un pronóstico estricto hacia adelante.
- El reporte de duplicados de `split_verificacion.txt` indica que 121 filas de ataque de validación y 33 de prueba tienen un vector de features idéntico al de alguna fila de entrenamiento. No se eliminan.
- El orden dentro de cada minuto es el del archivo y no es cronológico; su efecto se mide en el ADR 0016.
