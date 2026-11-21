# Detección de patrones de ataque lentos y distribuidos utilizando redes neuronales recurrentes: Módulo Gated Recurrent Unit

Trabajo de graduación — Universidad del Valle de Guatemala, Facultad de Ingeniería.

## Autor

| Nombre | Carné |
|---|---|
| Esteban Zambrano Garoz | 22119 |

El preprocesamiento de datos es compartido con los trabajos de Juan Pablo Solís Albizurez (22102, módulo SVM) y Edwin Andrés Ortega Kou (22305, módulo XGBoost).

## Descripción

Sistema de detección de intrusiones basado en una red neuronal recurrente **GRU** para identificar ataques de denegación de servicio **Low and Slow** (Slowloris y Slowhttptest) en flujos de red del dataset **CIC-IDS-2017** (miércoles, `GeneratedLabelledFlows`).

El flujo de trabajo incluye:

- **Limpieza** de los flujos: reconstrucción del `Timestamp` a 24 h, eliminación de atajos de laboratorio (puertos, protocolo) y de filas mal etiquetadas.
- **Partición cronológica** por bloques temporales (70/15/15), sin fuga de información entre entrenamiento, validación y prueba.
- **Escalado** con transformación logarítmica con signo y `StandardScaler` ajustado solo con entrenamiento.
- **Secuencias** deslizantes de W = 10 flujos, que no cruzan bloques ni particiones.
- **Modelo** `GRU(32) → Dropout(0.3) → Dense(1, sigmoid)`, entrenado con 5 semillas y umbral elegido en validación (F2 máximo).
- **Ablaciones** (W = 1, y orden aleatorio frente al orden del CSV), **evaluación única en prueba** y **análisis de errores**.

Resultado del modelo principal en prueba (66,454 flujos, media ± desviación estándar de 5 semillas): **recall 0.973 ± 0.011**, **F1 0.933 ± 0.011**.

## Tecnologías

- Python 3
- TensorFlow / Keras
- scikit-learn
- NumPy y pandas
- Matplotlib
- Jupyter Notebook
- Streamlit (demo interactiva)

## Requisitos previos

- Python 3.12 o 3.13 (los notebooks del GRU se ejecutaron con Python 3.13.3)
- `pip` y `venv`
- Git
- Unos 8 GB de espacio libre en disco (dataset y secuencias `.npy`)
- No se requiere GPU: el entrenamiento se realizó en CPU.
- Dataset **CIC-IDS-2017**, archivo `GeneratedLabelledFlows.zip`, disponible en <https://www.unb.ca/cic/datasets/ids-2017.html>. No se incluye en el repositorio por su tamaño.

## Instalación

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/GraduacionesUVG/2026-22119-zambrano-garoz-esteban.git
   cd 2026-22119-zambrano-garoz-esteban/src
   ```

2. Crear y activar un entorno virtual:

   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   ```

3. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Copiar el archivo de variables de entorno (opcional; el proyecto no usa credenciales):

   ```bash
   cp .env.example .env
   ```

5. Descomprimir `GeneratedLabelledFlows.zip` y colocar el CSV del miércoles en `src/data/raw/` (ruta esperada: `src/data/raw/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv`).

## Ejecución

Desde `src/`, con el entorno virtual activo:

```bash
jupyter notebook
```

Ejecutar los notebooks en este orden:

| Orden | Notebook | Qué hace |
|---|---|---|
| 1 | `notebooks/Cleaning/limpieza_low_and_slow.ipynb` | Limpieza y partición cronológica |
| 2 | `notebooks/Cleaning/Analisis_Exploratorio_actualizado.ipynb` | Análisis exploratorio |
| 3 | `python scripts/gru/build_gru_sequences.py` (desde `src/`) | Genera las secuencias W = 10 en `data/sequences/` |
| 4 | `notebooks/gru/01_baseline.ipynb` | Entrenamiento de la GRU W = 10 (5 semillas) |
| 5 | `notebooks/gru/02_ablaciones.ipynb` | Ablaciones: W = 1 y orden aleatorio |
| 6 | `notebooks/gru/03_evaluacion_test.ipynb` | Evaluación única en prueba |
| 7 | `notebooks/gru/04_analisis_errores.ipynb` | Análisis de errores persistentes |

Los modelos entrenados están en `src/models/gru/` y los resultados (tablas, figuras y JSON) en `src/results/gru/`. Para revisar los resultados no es necesario volver a entrenar, pero sí tener el dataset, ejecutar la limpieza (paso 1) y generar las secuencias (paso 3): los notebooks 03 y 04 cargan los modelos y probabilidades guardados, y también las secuencias de `data/sequences/`.

### Demo interactiva

Desde `src/`, con el entorno virtual activo:

```bash
streamlit run demo/app.py
```

Hace inferencia con el modelo GRU entrenado sobre el conjunto de prueba, sin reentrenar.

## Capturas

Capturas del sistema en funcionamiento, en [`/capturas`](capturas/):

| # | Archivo | Contenido |
|---|---|---|
| 1 | [captura-01.png](capturas/captura-01.png) | Demo: pantalla principal (configuración e instrucciones) |
| 2 | [captura-02.png](capturas/captura-02.png) | Demo: métricas de la inferencia en prueba |
| 3 | [captura-03.png](capturas/captura-03.png) | Demo: matriz de confusión, distribución de clases y errores |
| 4 | [captura-04.png](capturas/captura-04.png) | Demo: curvas ROC y precision-recall |
| 5 | [captura-05.png](capturas/captura-05.png) | Demo: recall por tipo de ataque y falsos positivos |
| 6 | [captura-06.png](capturas/captura-06.png) | Demo: probabilidad de ataque a lo largo del test |
| 7 | [captura-07.png](capturas/captura-07.png) | Demo: inspección de una ventana de 10 flujos |
| 8 | [captura-08.png](capturas/captura-08.png) | Verificación del split cronológico por bloques |
| 9 | [captura-09.png](capturas/captura-09.png) | Baseline GRU W = 10: resumen por semilla en validación |
| 10 | [captura-10.png](capturas/captura-10.png) | Ablaciones: tabla comparativa en validación |
| 11 | [captura-11.png](capturas/captura-11.png) | Ablaciones: conclusiones sobre el orden del CSV |

## Demo

Video demostrativo: [`/demo/demo.mp4`](demo/demo.mp4)

## Documentación

- Informe final: [`/docs/informe_final.pdf`](docs/informe_final.pdf)
- Consentimiento de uso de foto y video firmado: [`/docs/consentimiento_firmado.pdf`](docs/consentimiento_firmado.pdf)
- Decisiones de diseño (ADR) y resultados detallados: `src/docs/gru/`
