# ADR 0013: Arquitectura y entrenamiento

- **Estado:** Aceptado
- **Fecha:** 2026-10-11
- **Reemplaza a:** ADR 0003 y la parte del GRU del ADR 0004 (archivados)

## Contexto

Se necesita una configuración única del GRU que sirva para todas las variantes de datos (ADR 0012 y ADR 0016), de modo que las diferencias entre resultados se deban a los datos y no al modelo. El dataset está desbalanceado: los ataques son el 2.5 % de las filas (ADR 0009).

## Decisión

**Arquitectura** (`build_gru_model()` en `src/models/gru/model.py`): `GRU(32) → Dropout(0.3) → Dense(1, sigmoid)`.

**Entrenamiento** (`train_gru_model()` en `src/models/gru/train.py`):

| Parámetro | Valor |
|---|---|
| Optimizador | Adam, tasa de aprendizaje 1e-3 |
| Pérdida | Entropía cruzada binaria |
| Tamaño de batch | 64 |
| Épocas máximas | 50 |
| Parada temprana | `val_loss`, paciencia de 10 épocas, se restauran los mejores pesos |
| Desbalance | `class_weight` balanceado, calculado sobre entrenamiento |

Estos valores constan en el campo `config` de los `*_seeds.json` de `results/gru/`. Con los datos de entrenamiento actuales, los pesos de clase son 0.514 para BENIGN y, para ataque, 18.86 en GRU W=10 (orden CSV) y GRU W=1 y 18.88 en GRU W=10 (orden aleatorio), que tiene 6 ataques menos en entrenamiento (`results/gru/00_datos/pesos_clase.json`).

La parada temprana vigila `val_loss` y no el recall. El recall de validación se usa después, para elegir el umbral (ADR 0015).

## Alternativas consideradas

- **Vigilar el recall de validación.** Descartada: el recall depende del umbral de 0.5 y de pocos ataques, mientras que `val_loss` usa las probabilidades de todas las filas de validación.
- **Paciencia menor.** Descartada: `val_loss` presenta picos entre épocas (conclusiones de `01_baseline.ipynb`), y una paciencia corta detendría el entrenamiento en un pico.
- **SMOTE sobre las secuencias.** Descartada: interpolar entre dos ventanas produce una secuencia que no corresponde a ningún tráfico real. `class_weight` no crea datos.
- **Otras arquitecturas o tamaños.** No se exploraron (ADR 0018).

## Consecuencias

- Ninguna corrida llegó a las 50 épocas: se entrenaron entre 11 y 46, y la mejor época estuvo entre la 1 y la 36 (`*_seeds.json`, campos `epocas_entrenadas` y `mejor_epoca`).
- Tiempo de entrenamiento por semilla en CPU (`results/gru/03_evaluacion_test/tiempo_entrenamiento.csv`): 265.7 ± 56.6 s en GRU W=10 (orden aleatorio), 153.2 ± 41.3 s en GRU W=10 (orden CSV) y 143.8 ± 56.4 s en GRU W=1.
- El modelo guardado es el de la mejor época según `val_loss`, no el de la última.
- SVM y XGBoost tratan el desbalance con sus propios mecanismos; esa decisión corresponde a sus responsables.
