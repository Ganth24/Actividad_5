# CHANGELOG - Actividad 5

## v1.0.0 - 2026-10-18

### Dataset y gobierno de datos
- Se incluye el archivo original íntegro de Palmer Penguins (344 filas, 8 columnas) con trazabilidad a palmerpenguins y copia empaquetada de plotnine.
- SHA-256 original: `1867a776a83379df4219f227bb1effb967da12adb13732127c8c8d120434c29b`.
- Se agregó `datos_prep.py` con verificación de esquema, tipos, normalización de valores faltantes, reglas de positividad, descarte de objetivo nulo, eliminación de duplicados exactos y orden estable.
- Dataset limpio: 344 filas y 8 columnas; **0 duplicados eliminados**, **0 etiquetas objetivo faltantes**. Persisten faltantes en variables predictoras, tratados dentro de cada fold.
- SHA-256 limpio: `9054e0bf9ac999173106073b31e4cbce12513642e911719f6b9a5dffce17638b`.

### Entrenamiento y evaluación
- Se añadió GridSearchCV estratificado de 5 folds, semilla 42, criterio F1 macro y evaluación independiente en prueba 20 % (69 registros).
- Se entrenaron dos pipelines: regresión logística (6 combinaciones de hiperparámetros) y Random Forest (16 combinaciones).
- Regresión logística: F1 macro prueba **1.0000**; Random Forest: **0.9829**.
- Se añadieron 2,000 remuestreos bootstrap pareados y gráficos EDA, matrices de confusión, curvas ROC y comparativa de métricas.
- Se añadieron serializaciones `.joblib`, resultados legibles JSON/CSV, notebook Colab y pruebas unitarias.

### Seguimiento y limitaciones verificadas
- Script preparado para registrar corridas finales y corridas hijas de candidatos Grid Search en MLflow, incluyendo modelos, hiperparámetros, métricas, hashes y artefactos.
- **No se ejecutó MLflow en el entorno de generación (paquete no disponible y sin conectividad para instalarlo).** Se ejecutó `python fuentes/train.py --sin-mlflow`, de forma explícita y sin simular corridas.
- **No se publicó a una cuenta de GitHub**; Git se inicializó localmente y las instrucciones de publicación se incluyen en `README.md`.
- Capturas genuinas MLflow pendientes de realizar tras ejecución real; instrucciones detalladas incluidas en `evidencias_mlflow/INSTRUCCIONES.md`.

### Riesgos abiertos y deuda técnica
- La muestra reducida y el muestreo ecológico localizado pueden generar métricas optimistas.
- Pendiente lockfile de dependencias, CI con MLflow disponible, validación por año/isla, tracking remoto persistente, vigilancia de drift y validación externa.
