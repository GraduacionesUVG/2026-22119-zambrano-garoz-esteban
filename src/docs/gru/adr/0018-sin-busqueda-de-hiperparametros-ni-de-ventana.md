# ADR 0018: Sin búsqueda de hiperparámetros ni de tamaño de ventana

- **Estado:** Aceptado
- **Fecha:** 2026-10-11
- **Reemplaza a:** ADR 0007 y ADR 0008 (archivados)

## Contexto

El protocolo de investigación (`docs/Protocolo_de_Investigación_GRU.pdf`) contempla ajustar hiperparámetros "cuando sea necesario" y evaluar otros tamaños de ventana "si el alcance y el tiempo lo permiten". Tras las ablaciones (ADR 0016) había que decidir si se hacía esa búsqueda antes de evaluar en prueba.

## Decisión

No se buscan hiperparámetros ni otros tamaños de ventana. Todas las variantes usan la configuración del ADR 0013 y W = 10 (o W = 1 en la ablación).

La decisión se tomó **después de las ablaciones (`02_ablaciones.ipynb`) y antes de la evaluación en prueba**.

### Razones

1. **Las ablaciones necesitan una configuración fija.** Mantener la misma configuración en todas las variantes es lo que permite que las ablaciones aíslen el efecto de la ventana y del orden.
2. **El factor que determina el desempeño es el orden de las secuencias.** Las dos ablaciones cumplieron la regla fijada de antemano (ADR 0016).
3. **El desempeño en validación ya era cercano al techo.** En GRU W=10 (orden aleatorio), el recall en el umbral F2 es de 0.989 ± 0.006 y la PR-AUC de 0.997 ± 0.001; en ese umbral, la desviación entre semillas del recall, de F2 y de la precision está entre 0.005 y 0.009 (`results/gru/02_ablaciones/aleatorio_w10_seeds.json`).
4. **El protocolo lo deja condicionado.** Indica ajustar hiperparámetros "cuando sea necesario" y evaluar otros tamaños de ventana "si el alcance y el tiempo lo permiten".

### Evidencia posterior

Lo siguiente se obtuvo después de la decisión y no la motivó. El análisis de errores (`04_analisis_errores.ipynb`, ADR 0020) muestra que los errores persistentes son flujos cuyas features se asemejan a las de la otra clase (`results/gru/04_analisis_errores/perfil_features.csv`):

- los falsos negativos tienen una duración mediana de 192 µs, frente a 8.5 s en los ataques detectados;
- los falsos positivos tienen una duración mediana de 1.7 s, frente a 31 ms en el tráfico benigno bien clasificado.

## Alternativas consideradas

- **Buscar hiperparámetros antes de las ablaciones.** Descartada: cada variante podría terminar con una configuración distinta y las ablaciones dejarían de aislar un solo factor.
- **Buscar después de evaluar en prueba.** Descartada: contaminaría la evaluación.
- **Evaluar W = 20 u otras ventanas.** Descartada por la razón 2: con orden aleatorio, la ventana de 10 flujos ya no supera al flujo individual en validación.

## Consecuencias

- La conclusión del trabajo se limita a la configuración evaluada: **con esta configuración, W=10 no supera a W=1** cuando el orden dentro del minuto se sortea.
- No se puede afirmar que otra arquitectura, otra ventana u otros hiperparámetros no aprovecharían la secuencia.
- La búsqueda de hiperparámetros y de ventana sobre la variante con orden aleatorio queda como **trabajo futuro**.
- `scripts/gru/build_gru_sequences.py` solo genera W = 10.
