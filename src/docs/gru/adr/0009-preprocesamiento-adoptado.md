# ADR 0009: Preprocesamiento adoptado por el GRU

- **Estado:** Aceptado
- **Fecha:** 2026-10-11
- **Reemplaza a:** ADR 0002 y la parte de datos del ADR 0001 (archivados)

## Contexto

El proyecto compara un GRU con SVM y XGBoost para detectar ataques Low & Slow (Slowloris y Slowhttptest) en la captura del miércoles 5/7/2017 de CIC-IDS2017. Para que la comparación sea válida, los tres modelos parten del mismo dataset limpio, que produce el código compartido de `src/preprocessing/clean.py` y el notebook `notebooks/Cleaning/limpieza_low_and_slow.ipynb`.

Este ADR registra qué preprocesamiento adopta el GRU. No decide el preprocesamiento compartido, que es responsabilidad del equipo.

## Decisión

El GRU usa como única entrada `data/processed/wednesday_low_and_slow_clean.csv`, cargado con `cargar_dataset_limpio()`. No aplica ninguna limpieza propia. Ese archivo es el resultado de `load_and_clean()`, que hace lo siguiente:

1. **Filtro de clases.** Conserva BENIGN, DoS slowloris y DoS Slowhttptest. Descarta DoS Hulk, DoS GoldenEye y Heartbleed, que están fuera del alcance del protocolo.
2. **Exclusión de respuestas de ataque.** Excluye los flujos BENIGN entre la interfaz NAT por la que entran los atacantes (172.16.0.1) y el servidor víctima (192.168.10.50). Son el lado servidor de conexiones de ataque etiquetado como benigno.
3. **Corrección del Timestamp.** El archivo trae la hora en reloj de 12 horas sin AM/PM. Las horas menores a 8 se desplazan 12 horas, porque la captura va de la mañana a la tarde.
4. **Limpieza básica.** Elimina filas con valores nulos o infinitos, duplicados exactos y filas con valores negativos en duración, tiempos entre paquetes o tasas.
5. **Columnas eliminadas.**
   - Identificadoras: `Flow ID`, `Source IP`, `Destination IP` y `Source Port`.
   - Atajos del entorno de laboratorio: `Destination Port` y `Protocol`. Todos los ataques van al puerto 80 y son TCP, así que estas columnas separan las clases sin describir el comportamiento lento.
   - Redundantes (8) y constantes (10), listadas en `COLUMNAS_REDUNDANTES` y `COLUMNAS_CONSTANTES`.
6. **Exclusión de ataques fuera de ventana.** Excluye las filas etiquetadas como ataque cuyo inicio queda fuera de la ventana documentada de ese ataque, con ±2 minutos de tolerancia.

Las filas excluidas en los pasos 2 y 6 no se reetiquetan, porque su clase real es desconocida.

### Cifras

| Dato | Valor | Fuente |
|---|---:|---|
| Filas de la captura del miércoles | 692,703 | Output de la sección 2 de `limpieza_low_and_slow.ipynb` |
| Filas tras el filtro de clases | 451,326 | Mismo output |
| Respuestas de ataque excluidas | 6,789 | Texto de la sección 3 del mismo notebook |
| Filas eliminadas por la limpieza básica | 917 | `results/gru/00_datos/conteos_limpieza.json` |
| Ataques excluidos por estar fuera de ventana | 6 (todos Slowloris) | Output de la sección 7 del mismo notebook |
| Filas del dataset limpio | 443,614 | Suma de `results/gru/split_verificacion.txt` |
| BENIGN | 432,325 | Suma de `split_verificacion.txt` |
| DoS slowloris | 5,790 | Suma de `split_verificacion.txt` |
| DoS Slowhttptest | 5,499 | Suma de `split_verificacion.txt` |
| Features | 59 | `FEATURE_COLUMNS` en `src/models/gru/sequences.py` |

## Alternativas consideradas

- **Limpieza propia del GRU.** Descartada: los tres modelos dejarían de compartir datos y la comparación perdería validez.
- **Conservar `Destination Port` y `Protocol`.** Descartada por el equipo: un modelo podría obtener un recall alto aprendiendo "puerto 80 = ataque". `basic_cleaning(eliminar_atajos=False)` existe para medir ese efecto como ablación, pero el GRU no la usa.
- **Reetiquetar las filas con ruido de etiqueta en lugar de excluirlas.** Descartada: no hay forma de confirmar su clase.

## Consecuencias

- `FEATURE_COLUMNS` debe coincidir con las columnas del CSV sin `Timestamp`, `Label` ni `label_binaria`. `scripts/gru/build_gru_sequences.py` lo comprueba y se detiene si no coinciden.
- Cualquier cambio en `clean.py` o en el CSV obliga a regenerar secuencias, modelos y resultados.
- El GRU no dispone de IPs ni de puertos, así que no puede agrupar flujos por host ni por conexión (ver ADR 0012 y ADR 0016).
- Los ataques son el 2.5 % del dataset (11,289 de 443,614), por lo que el entrenamiento compensa el desbalance (ADR 0013) y la accuracy no se usa como métrica principal.
