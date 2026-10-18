# Evidencias auténticas de MLflow - pasos para completar

**Estado:** PENDIENTE DE EJECUCION REAL. Esta entrega no contiene capturas de una interfaz de MLflow porque la librería no pudo instalarse en el entorno offline de generación. Las gráficas de `figuras/` SÍ provienen de un entrenamiento real de scikit-learn, pero NO son capturas de MLflow.

## Pasos

1. Desde la carpeta `Actividad5` y con acceso a PyPI: `python -m pip install -r requirements.txt`.
2. Ejecuta `python fuentes/train.py` **sin** `--sin-mlflow` para registrar todas las corridas; no basta con volver a correr el CSV ni únicamente el notebook offline.
3. Ejecuta `mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000`.
4. Abre `http://127.0.0.1:5000` y el experimento `Actividad5_Palmer_Penguins_Clasificacion`.
5. Toma capturas reales (sin editar su contenido), guárdalas como:
   - `capturas/01_experimento_corridas.png`: vista de corridas, nombres y métricas F1.
   - `capturas/02_parametros_metricas.png`: detalle de `final_regresion_logistica` o `final_random_forest`, hiperparámetros y métricas.
   - `capturas/03_artefactos_modelo.png`: detalle de artefactos/modelo de alguna corrida final y/o ejemplos de corridas hijas `candidato_...`.
6. Verifica que hay exactamente al menos 2 corridas finales, 22 corridas hijas (6+16) y 1 corrida comparativa cuando se ejecuta una sola vez, además de modelos y archivos asociados. Algunas versiones de la UI agrupan y filtran runs por defecto; ajusta filtros si no aparecen.
7. Añade las imágenes a Git, confirma el commit y vuelve a generar el ZIP actualizado. Opcional: exporta `mlruns/` en un ZIP separado para preservar la trazabilidad de un FileStore local (incluye modelos/pesos y metadatos).

**No confundas captura preparada o reconstruida con evidencia de tracking realmente ejecutado.**
