"""Clasificacion multiclase con GridSearchCV, prueba holdout y MLflow.

Por defecto usa MLflow; --sin-mlflow permite comprobar entrenamiento en
entornos offline sin simular que se ha registrado un experimento.
"""
from __future__ import annotations

import argparse
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import t
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize

try:
    from .datos_prep import CATEGORICAL, FEATURES, NUMERICAL, TARGET, preparar_dataset, sha256_file
except ImportError:
    from datos_prep import CATEGORICAL, FEATURES, NUMERICAL, TARGET, preparar_dataset, sha256_file

SEED = 42
EXPERIMENT_NAME = "Actividad5_Palmer_Penguins_Clasificacion"


def pipeline_modelo(nombre: str) -> Pipeline:
    numeric_steps = [("imputar", SimpleImputer(strategy="median"))]
    if nombre == "regresion_logistica":
        numeric_steps.append(("escalar", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    cat_pipe = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numerico", numeric_pipe, NUMERICAL),
        ("categorico", cat_pipe, CATEGORICAL),
    ], remainder="drop")
    if nombre == "regresion_logistica":
        model = LogisticRegression(max_iter=3000, random_state=SEED)
    elif nombre == "random_forest":
        model = RandomForestClassifier(random_state=SEED, n_jobs=1)
    else:
        raise ValueError(nombre)
    return Pipeline([("preprocesar", preprocess), ("modelo", model)])


def rejillas() -> dict:
    return {
        "regresion_logistica": {
            "modelo__C": [0.1, 1.0, 10.0],
            "modelo__class_weight": [None, "balanced"],
        },
        "random_forest": {
            "modelo__n_estimators": [100, 200],
            "modelo__max_depth": [None, 6],
            "modelo__min_samples_leaf": [1, 3],
            "modelo__class_weight": [None, "balanced"],
        },
    }


def metricas(y_true: pd.Series, pred: np.ndarray, prob: np.ndarray, classes: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_macro": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "roc_auc_ovr_macro": float(roc_auc_score(y_true, prob, labels=classes, multi_class="ovr", average="macro")),
    }


def bootstrap_pareado(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, reps: int = 2000) -> dict:
    """IC exploratorio sobre diferencia F1-macro (RF - LogReg), remuestreo estratificado.

    Ambos modelos usan los mismos casos de prueba en cada replica.
    Las replicas no son replicaciones de entrenamiento y NO justifican p-values.
    """
    rng = np.random.default_rng(SEED)
    grupos = [np.flatnonzero(y == value) for value in np.unique(y)]
    deltas = np.empty(reps, dtype=float)
    for i in range(reps):
        sample = np.concatenate([rng.choice(ids, size=len(ids), replace=True) for ids in grupos])
        deltas[i] = f1_score(y[sample], pred_b[sample], average="macro") - f1_score(y[sample], pred_a[sample], average="macro")
    return {
        "method": "bootstrap pareado estratificado de casos de test",
        "resamples": reps,
        "seed": SEED,
        "difference": "F1_macro(random_forest) - F1_macro(regresion_logistica)",
        "point_difference": float(f1_score(y, pred_b, average="macro") - f1_score(y, pred_a, average="macro")),
        "ci95_percentile": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "interpretation_limit": "Descriptivo; no atribuir significacion estadistica causal ni generalizacion poblacional por el test pequeno.",
    }


def grafica_eda(df: pd.DataFrame, figdir: Path) -> dict:
    import seaborn as sns
    figdir.mkdir(parents=True, exist_ok=True)
    counts = df[TARGET].value_counts().reindex(["Adelie", "Chinstrap", "Gentoo"])
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.3))
    sns.countplot(data=df, x="species", order=counts.index, hue="species", legend=False, ax=axes[0], palette="deep")
    axes[0].set(title="Distribucion por especie", xlabel="Especie", ylabel="Registros")
    sns.scatterplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species", ax=axes[1], alpha=.8)
    axes[1].set(title="Morfologia del pico", xlabel="Longitud del pico (mm)", ylabel="Profundidad del pico (mm)")
    fig.tight_layout(); fig.savefig(figdir / "01_eda_especies_pico.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.1))
    nulls = df.isna().sum(); nulls = nulls[nulls > 0].sort_values(ascending=False)
    if len(nulls):
        axes[0].bar(nulls.index, nulls.values)
        axes[0].tick_params(axis="x", labelrotation=40)
    axes[0].set(title="Datos faltantes antes de imputar", ylabel="Celdas")
    sns.boxplot(data=df, x="species", y="body_mass_g", hue="species", legend=False, ax=axes[1], palette="deep")
    axes[1].set(title="Masa corporal por especie", xlabel="Especie", ylabel="Masa (g)")
    fig.tight_layout(); fig.savefig(figdir / "02_nulos_y_masa.png", dpi=160); plt.close(fig)
    iqr_flags = {}
    for c in NUMERICAL:
        q1, q3 = df[c].quantile([.25,.75])
        iqr = q3 - q1
        flags = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
        iqr_flags[c] = int(flags)
    return {"class_balance": {str(k): int(v) for k, v in counts.items()}, "iqr_outlier_flags": iqr_flags,
            "iqr_policy": "Banderas exploratorias, no se eliminan extremos fisiologicos plausibles sin evidencia de error."}


def grafica_roc(model: Pipeline, X: pd.DataFrame, y: pd.Series, path: Path) -> None:
    classes = model.classes_
    probs = model.predict_proba(X)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for i, klass in enumerate(classes):
        truth_binary = (y.to_numpy() == klass).astype(int)
        fpr, tpr, _ = roc_curve(truth_binary, probs[:, i])
        ax.plot(fpr, tpr, label=str(klass))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax.set(title="Curvas ROC One-vs-Rest", xlabel="Tasa de falsos positivos", ylabel="Sensibilidad")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def grafica_confusion(model: Pipeline, X: pd.DataFrame, y: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.7, 4.5))
    ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusion (test sin usar en el ajuste)")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def guardar_comparacion(resultados: dict, figdir: Path) -> None:
    names = list(resultados)
    labels = ["Regresion logistica" if n == "regresion_logistica" else "Random forest" for n in names]
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro"]
    vals = np.array([[resultados[n]["test_metrics"][m] for m in metrics] for n in names])
    x = np.arange(len(metrics)); width = .34
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    for i in range(len(names)):
        ax.bar(x + (i-.5)*width, vals[i], width, label=labels[i])
    ax.set_ylim(0, 1.1); ax.set_ylabel("Puntuacion (0-1)")
    ax.set_xticks(x, ["Accuracy", "Precision\nmacro", "Recall\nmacro", "F1\nmacro", "ROC-AUC\nOvR macro"])
    ax.legend(ncol=2, loc="lower right")
    ax.set_title("Resultados finales en test (mismos 69 registros)")
    for i in range(len(names)):
        for j in range(len(metrics)):
            ax.text(x[j]+(i-.5)*width, vals[i,j]+.015, f"{vals[i,j]:.3f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "05_comparacion_modelos.png", dpi=160); plt.close(fig)


def ejecutar_experimento(repo_root: Path, enable_mlflow: bool = True) -> dict:
    repo_root = Path(repo_root).resolve()
    import_mlflow = None
    if enable_mlflow:
        try:
            import mlflow
            import mlflow.sklearn
            from mlflow.models import infer_signature
            import_mlflow = mlflow
        except ImportError as exc:
            raise RuntimeError("MLflow no instalado. Instala requirements.txt, o ejecuta --sin-mlflow para validar solo el entrenamiento; este modo NO genera evidencia MLflow.") from exc

    prep = preparar_dataset(repo_root)
    df = pd.read_csv(repo_root / "datos/datos_limp/penguins_limpios.csv")
    figdir = repo_root / "figuras"; figdir.mkdir(parents=True, exist_ok=True)
    resdir = repo_root / "resultados"; resdir.mkdir(parents=True, exist_ok=True)
    eda = grafica_eda(df, figdir)
    X = df[FEATURES]; y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    outcomes, predictions = {}, {}
    if import_mlflow is not None:
        # FileStore permite seguimiento local sin autenticacion; cada estudiante
        # configura su URI basada en la raiz actual, sin rutas hardcodeadas.
        tracking_path = (repo_root / "mlruns").resolve()
        tracking_path.mkdir(exist_ok=True)
        import_mlflow.set_tracking_uri(tracking_path.as_uri())
        import_mlflow.set_experiment(EXPERIMENT_NAME)
    for nombre, params in rejillas().items():
        search = GridSearchCV(
            estimator=pipeline_modelo(nombre), param_grid=params,
            scoring={"f1_macro":"f1_macro", "accuracy":"accuracy"},
            refit="f1_macro", cv=cv, return_train_score=True,
            n_jobs=1, error_score="raise",
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        pred = best.predict(X_test)
        prob = best.predict_proba(X_test)
        scores = metricas(y_test, pred, prob, best.classes_)
        candidates = pd.DataFrame(search.cv_results_)
        cv_file = resdir / f"gridsearch_{nombre}.csv"
        candidates.to_csv(cv_file, index=False)
        path_cm = figdir / f"03_confusion_{nombre}.png"
        path_roc = figdir / f"04_roc_{nombre}.png"
        grafica_confusion(best, X_test, y_test, path_cm)
        grafica_roc(best, X_test, y_test, path_roc)
        joblib.dump(best, resdir / f"modelo_{nombre}.joblib")
        outcomes[nombre] = {
            "algorithm": nombre,
            "train_rows": int(len(y_train)), "test_rows": int(len(y_test)),
            "test_metrics": scores,
            "cv_f1_mean": float(search.cv_results_["mean_test_f1_macro"][search.best_index_]),
            "cv_f1_std": float(search.cv_results_["std_test_f1_macro"][search.best_index_]),
            "best_params": search.best_params_,
            "grid_candidates": int(len(candidates)),
            "cv_folds": 5,
            "model_file": f"resultados/modelo_{nombre}.joblib",
        }
        predictions[nombre] = pred
        if import_mlflow is not None:
            mlflow = import_mlflow
            with mlflow.start_run(run_name=f"final_{nombre}"):
                mlflow.set_tags({"algoritmo":nombre, "dataset":"Palmer Penguins", "rol":"modelo_final", "seed":str(SEED),
                    "raw_sha256":prep["raw_sha256"], "clean_sha256":prep["clean_sha256"],
                    "split":"holdout_80_20_stratified", "seleccion":"grid_search_cv_solo_train"})
                mlflow.log_params({"n_splits":5, "test_size":.2,"random_state":SEED,
                    "num_candidatos":len(candidates), "criterio_optimizacion":"f1_macro"})
                mlflow.log_params({k: str(v) for k,v in search.best_params_.items()})
                mlflow.log_metrics({"cv_f1_macro_mean":outcomes[nombre]["cv_f1_mean"],
                    "cv_f1_macro_std":outcomes[nombre]["cv_f1_std"],
                    **{f"test_{k}":v for k,v in scores.items()}})
                for artifact in [cv_file, path_cm, path_roc,
                    repo_root / "datos/datos_limp/auditoria_preparacion.json"]:
                    mlflow.log_artifact(str(artifact), artifact_path="evidencia")
                # Ejemplo y firma para reproducibilidad de la interfaz del modelo.
                signature = infer_signature(X_train.head(5), best.predict(X_train.head(5)))
                mlflow.sklearn.log_model(sk_model=best, name="modelo_pipeline", signature=signature,
                    input_example=X_train.head(5))
                # Se anidan los hiperparametros probados, con su resultado CV.
                for i, row in candidates.iterrows():
                    with mlflow.start_run(run_name=f"candidato_{i+1:02d}_{nombre}", nested=True):
                        mlflow.set_tags({"algoritmo": nombre, "rol": "candidato_cv"})
                        mlflow.log_params({k: str(v) for k, v in row["params"].items()})
                        mlflow.log_metrics({"cv_f1_macro_mean":float(row["mean_test_f1_macro"]),
                            "cv_f1_macro_std":float(row["std_test_f1_macro"]),
                            "cv_accuracy_mean":float(row["mean_test_accuracy"])})
    comparison = bootstrap_pareado(y_test.to_numpy(), predictions["regresion_logistica"], predictions["random_forest"])
    guardar_comparacion(outcomes, figdir)
    comparison_path = resdir / "comparacion_estadistica.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "tracking_mlflow_ejecutado":bool(import_mlflow is not None),
        "experiment_name":EXPERIMENT_NAME, "seed": SEED,
        "dataset": {"original_rows":prep["original_rows"], "clean_rows":prep["final_rows"],
                    "raw_sha256":prep["raw_sha256"], "clean_sha256":prep["clean_sha256"]},
        "environment": {"python":platform.python_version(), "scikit_learn":sklearn.__version__,
            "pandas":pd.__version__, "numpy":np.__version__,
            "mlflow":import_mlflow.__version__ if import_mlflow is not None else "NO_INSTALADO"},
        "training": {"rows":len(y_train), "class_counts": {str(k):int(v) for k,v in y_train.value_counts().items()}},
        "test": {"rows":len(y_test), "class_counts":{str(k):int(v) for k,v in y_test.value_counts().items()}},
        "eda":eda, "models":outcomes, "statistical_comparison":comparison,
    }
    (resdir / "resultados_experimento.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"modelo":k,**v["test_metrics"], "cv_f1_mean":v["cv_f1_mean"], "cv_f1_std":v["cv_f1_std"]} for k,v in outcomes.items()]).to_csv(resdir / "metricas_comparativas.csv", index=False)
    if import_mlflow is not None:
        with import_mlflow.start_run(run_name="resumen_comparativo"):
            import_mlflow.set_tags({"rol":"comparacion", "raw_sha256":prep["raw_sha256"], "clean_sha256":prep["clean_sha256"]})
            for fname in ["comparacion_estadistica.json", "metricas_comparativas.csv", "resultados_experimento.json"]:
                import_mlflow.log_artifact(str(resdir / fname), artifact_path="resumen")
            import_mlflow.log_artifact(str(figdir / "05_comparacion_modelos.png"), artifact_path="graficos")
            import_mlflow.log_metric("delta_f1_rf_vs_logreg", comparison["point_difference"])
    return report


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento, GridSearchCV y registro MLflow")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--sin-mlflow", action="store_true", help="Validacion offline: NO registra experimentos en MLflow")
    args = parser.parse_args()
    data = ejecutar_experimento(args.repo_root, enable_mlflow=not args.sin_mlflow)
    for model, values in data["models"].items():
        print(model, json.dumps(values["test_metrics"], ensure_ascii=False))
    print("MLflow registrado:", data["tracking_mlflow_ejecutado"])

if __name__ == "__main__":
    main()
