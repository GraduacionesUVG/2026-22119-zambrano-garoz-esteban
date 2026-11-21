# ADR archivados (pipeline anterior)

Esta carpeta contiene los ADR 0001–0008 y `mecanismo-mapeo-secuencias.md`, escritos para el pipeline del GRU anterior a la corrección del preprocesamiento compartido.

Se archivaron porque sus cifras, sus rutas y parte de sus decisiones ya no corresponden al código ni a los resultados actuales. Se conservan como registro histórico y no deben citarse como documentación vigente. Los modelos, notebooks y resultados de ese pipeline están en `models/gru/_v0_pre_fix/`, `notebooks/gru/_v0_pre_fix/` y `results/gru/_v0_pre_fix/`.

Las decisiones vigentes están en los ADR 0009–0021 de `docs/gru/adr/`.

## Equivalencias

| Archivado | Tema | ADR vigente |
|---|---|---|
| 0001 | Split cronológico por bloques y escalado | 0009 (datos), 0010 (split), 0011 (escalado) |
| 0002 | Columnas constantes | 0009 |
| 0003 | Criterio de EarlyStopping | 0013 |
| 0004 | Balanceo de clases | 0013 (parte del GRU) |
| 0005 | Umbral de decisión | 0015 (umbral), 0019 (evaluación en test) |
| 0006 | Análisis de falsos negativos | 0020 |
| 0007 | Exploración de hiperparámetros | 0018 |
| 0008 | Comparación de ventanas | 0016 (ablación de ventana), 0018 |
| `mecanismo-mapeo-secuencias.md` | Mapeo de secuencia a fila original | 0012 |
