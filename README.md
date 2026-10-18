# Actividad 5 - MLOps reproducible con Palmer Penguins

**Clasificación multiclase de pingüinos Adelie, Chinstrap y Gentoo con regresión logística y Random Forest.** Incluye versiones del dataset, preparación auditada, Grid Search con 5 folds, evaluación final, modelos serializados, gráficos y registro MLflow configurable.

> **Estado verificable de esta entrega:** el entrenamiento, las figuras, los archivos de métricas y las pruebas de código se ejecutaron en entorno offline. **MLflow NO se ejecutó en el entorno de generación**, porque su paquete no estaba disponible y no había acceso a PyPI; por ello, los resultados locales no deben confundirse con corridas MLflow. El script implementa el registro real al ejecutarse con las dependencias instaladas. **La publicación en una cuenta de GitHub y las capturas auténticas del panel requieren acceso a esa cuenta y una instalación de MLflow.**

## 1. Dataset y procedencia

- Datos públicos: [palmerpenguins (Allison Horst et al.)](https://github.com/allisonhorst/palmerpenguins), archivo `inst/extdata/penguins.csv`.
- Fuente de la copia utilizada: `plotnine/data/penguins.csv` (distribución instalada `plotnine` 0.15.3); misma estructura del dataset público de 344 filas y 8 columnas.
- Objetivo: `species` (Adelie, Chinstrap, Gentoo).
- Observaciones originales: 344. No se detectaron duplicados exactos ni etiquetas nulas.
- **SHA-256 CSV original:** `1867a776a83379df4219f227bb1effb967da12adb13732127c8c8d120434c29b`.
- **SHA-256 CSV limpio:** `9054e0bf9ac999173106073b31e4cbce12513642e911719f6b9a5dffce17638b`.
- Lectura y validación reproducibles sin conexión: el CSV original está incluido y no depende de descargas durante el entrenamiento.

### Diccionario de datos

| Campo | Tipo | Rol / significado |
|---|---|---|
| `species` | Categórico | Objetivo: especie de pingüino (tres categorías) |
| `island` | Categórico | Isla de observación (Biscoe, Dream, Torgersen) |
| `bill_length_mm` | Numérico | Longitud del pico en milímetros |
| `bill_depth_mm` | Numérico | Profundidad del pico en milímetros |
| `flipper_length_mm` | Numérico | Longitud de aleta en milímetros |
| `body_mass_g` | Numérico | Masa corporal en gramos |
| `sex` | Categórico | Sexo registrado: male, female o faltante |
| `year` | Entero | Año de observación; conserva la trazabilidad, **no** se usa como predictor |

La muestra no es una representación global de especies; procede de observaciones de Palmer Station (Antártida) durante 2007-2009. Se excluye `year` como predictor porque no es una propiedad biológica estable y podría inducir asociaciones de muestreo.

## 2. Estructura

```text
Actividad5/
├── datos/
│   ├── datos_ini/penguins_original.csv
│   └── datos_limp/
│       ├── penguins_limpios.csv
│       └── auditoria_preparacion.json
├── fuentes/
│   ├── __init__.py
│   ├── entrena.ipynb
│   ├── datos_prep.py
│   └── train.py
├── figuras/                  # exploración, matrices, ROC y comparativa
├── resultados/               # grid searches, métricas JSON/CSV, modelos joblib
├── evidencias_mlflow/
│   ├── capturas/              # pendiente: capturas reales del servidor MLflow
│   └── INSTRUCCIONES.md
├── reportes/reporte_tecnico.pdf
├── tests/test_pipeline.py
├── .github/workflows/pruebas.yml
├── .gitignore
├── requirements.txt
├── README.md
└── CHANGELOG.md
```

## 3. Reproducir de principio a fin

Desde la carpeta `Actividad5` con Python 3.11-3.13 y conectividad inicial para instalar dependencias:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q tests
python fuentes/datos_prep.py
python fuentes/train.py
```

`train.py` ejecuta **por defecto MLflow real** en `./mlruns/` (FileStore local) y crea el experimento `Actividad5_Palmer_Penguins_Clasificacion`. Cada modelo genera una corrida final; sus candidatos de Grid Search se registran en corridas hijas, además de una corrida `resumen_comparativo`. Abre otro terminal:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Abre `http://127.0.0.1:5000`, selecciona el experimento, y comprueba etiquetas, parámetros, métricas y artefactos del modelo. **No publiques este servidor sin autenticación; está pensado para uso local.**

Si faltara MLflow por estar sin internet, **solo para validar el entrenamiento sin fingir registro**:

```bash
python fuentes/train.py --sin-mlflow
```

Esa bandera reproduce los resultados y genera archivos en `resultados/` y `figuras/`, pero **no constituye evidencia MLflow**.

### Ejecutar el notebook en Colab

1. Sube el ZIP, descomprímelo en `/content/` y entra en la carpeta `Actividad5`, o clona el repositorio una vez que lo publiques.
2. Abre `fuentes/entrena.ipynb` en Google Colab.
3. Ejecuta las celdas desde la raíz `Actividad5`; la primera instala dependencias, y la celda principal llama a `ejecutar_experimento(..., enable_mlflow=True)`.
4. El tracking en Colab está en el sistema de archivos de la sesión. Guarda `mlruns/` o usa almacenamiento persistente externo si necesitas conservar las corridas.
5. Para visualizar la interfaz local desde tu PC, es más sencillo ejecutar la versión descargada mediante `mlflow ui` (no exponer Colab sin autenticación).

### Publicar en GitHub

El ZIP incluye un repositorio Git inicializado con historial local. **No existe en este momento un repositorio remoto publicado en tu cuenta.** Crea uno nuevo y vacío en GitHub, por ejemplo `Actividad5-MLflow`, y desde `Actividad5/` ejecuta:

```bash
git remote add origin https://github.com/TU_USUARIO/Actividad5-MLflow.git
git push -u origin main
```

Sustituye `TU_USUARIO` por tu usuario real y autentícate con el mecanismo de GitHub adecuado (p. ej. Git Credential Manager/SSH). No se incluyen credenciales.

## 4. Preparación, ausencia de fuga y reglas de negocio

- Valida las ocho columnas, normaliza espacios y marcas de nulos (`NA`, `N/A`, `?`, cadena vacía), convierte numéricos y año, convierte números **no positivos** en faltantes y restringe año a 2007-2009; no inventa valores.
- Descarta filas sin especie, elimina duplicados exactos (0 hallados en el dataset original), ordena de manera estable y registra auditoría con recuentos y SHA-256.
- El CSV limpio **puede mantener faltantes**. Para evitar filtraciones, cada fold aprende su imputación (mediana en variables numéricas; categoría más frecuente en categóricas) sobre sus propios datos de entrenamiento; después aplica One-Hot-Encoding (`handle_unknown='ignore'`). La regresión logística utiliza además `StandardScaler` dentro del pipeline. Random Forest no requiere escalar.
- Predictores: cuatro medidas corporales, `island` y `sex`; se excluye `year`. `species` es exclusivamente la etiqueta.
- Gráficas EDA: frecuencia por especie, relación longitud-profundidad del pico, datos nulos, masa por especie. El criterio IQR (1.5 × IQR) se usa solo para **señalización exploratoria**; 0 observaciones fueron marcadas por columna en esta copia; no se eliminan posibles ejemplares válidos sin criterio biológico.

## 5. Entrenamiento, validación y métricas

- Partición estratificada fija: entrenamiento 275 (80 % aprox.), prueba 69 (20 % aprox.). Semilla 42.
- `StratifiedKFold` de 5 pliegues, barajado con semilla 42. **Solo entrenamiento** participa en GridSearchCV. El test no interviene en la selección de modelos.
- Optimización: `f1_macro`, robusta ante el desbalance de clases. Se recopila `accuracy` en CV; en test se reportan `accuracy`, precisión, exhaustividad/recall, `F1` macro y ROC-AUC One-vs-Rest macro.
- Regresión logística: `C` ∈ {0.1, 1, 10}, `class_weight` ∈ {None, balanced} = **6 candidatos × 5 folds**. Justificación: controlar regularización y desbalance.
- Random Forest: 100/200 árboles, profundidad ilimitada/6, hojas mínimas 1/3 y `class_weight` None/balanced = **16 candidatos × 5 folds**. Justificación: variación de complejidad, varianza y atención al desbalance.
- La variabilidad CV es la **desviación estándar descriptiva** entre folds, no un intervalo de confianza.
- Se calcula diferencia de F1 macro test (`Random Forest - Regresión logística`) por **2,000 bootstrap pareados, estratificados por especie**; es una estimación exploratoria, sin convertir los folds dependientes en muestras independientes.

## 6. Resultados locales verificados (sin MLflow en esta ejecución)

| Modelo | F1 CV media ± desviación | Accuracy test | Precision macro | Recall macro | F1 macro test | ROC-AUC OvR macro |
|---|---:|---:|---:|---:|---:|---:|
| Regresión logística | 0.9970 ± 0.0060 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 0.9786 ± 0.0191 | 0.9855 | 0.9778 | 0.9889 | 0.9829 | 1.0000 |

Parámetros ganadores: regresión logística `C=10, class_weight=None`; Random Forest `n_estimators=100, max_depth=None, min_samples_leaf=1, class_weight=None`. La diferencia F1 macro (`RF - RL`) fue -0.0171 con intervalo bootstrap percentil 95 % [-0.0498, 0.0000]. En estos mismos 69 casos, la regresión logística fue ligeramente superior, sin evidencia para afirmar dominancia universal. Un ROC-AUC de 1.0 no implica clasificación 100 % correcta en un umbral concreto.

## 7. Evidencias MLflow y reproducibilidad

- **Resultados ejecutados**: `resultados/resultados_experimento.json`, `metricas_comparativas.csv`, ambos `gridsearch_*.csv`, archivos `.joblib`, y 7 gráficas PNG.
- **Código MLflow listo pero NO ejecutado en este entorno**: `fuentes/train.py` crea corridas y anida candidatos; registra `best_params`, métricas CV/test, hash de datos, modelo/pipeline, firma, ejemplo, matrices, curvas y cuadrículas.
- **Evidencia visual real pendiente**: `evidencias_mlflow/INSTRUCCIONES.md` indica las 3 capturas que debes obtener después de correr `python fuentes/train.py` con MLflow instalado. No se han inventado imágenes del panel.
- Versionamiento: commits Git locales y `CHANGELOG.md`; hashes SHA-256 de datos original y limpio; IDs de corrida consultables en MLflow cuando se ejecute.

## 8. Limitaciones, ética y deuda técnica

El conjunto es pequeño, no garantiza independencia entre observaciones de un mismo período de muestreo y contiene asociación entre isla y especie; las puntuaciones cercanas a 1 no demuestran transferencia fuera de Palmer Station. Mejora MLOps: bloquear versiones exactas de dependencias, integrar tests y checks de calidad de datos en CI, almacenar artefactos persistentes, registrar linaje del conjunto, probar evaluación temporal/geográfica (fuera de distribución), revisar drift y definir un proceso de aprobación antes de producción. No usar para decisiones conservacionistas sin validación externa.

## Fuentes y referencias

1. Horst, A. M., Hill, A. P. y Gorman, K. B. Repositorio [palmerpenguins](https://github.com/allisonhorst/palmerpenguins).
2. [Dataset CSV público de palmerpenguins](https://github.com/allisonhorst/palmerpenguins/blob/main/inst/extdata/penguins.csv).
3. [Documentación scikit-learn: GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
4. [Documentación MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/).
