"""Preparacion determinista de Palmer Penguins, sin fuga de informacion.

No se imputan los datos globalmente: la imputacion y la codificacion se
ajustan solo en particiones de entrenamiento dentro de Pipeline/GridSearchCV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "species"
CATEGORICAL = ["island", "sex"]
NUMERICAL = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
FEATURES = NUMERICAL + CATEGORICAL
COLUMNS = ["species", "island", *NUMERICAL, "sex", "year"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def limpiar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Valida esquema, normaliza cadenas, tipa, invalida imposibles y deduplica.

    Mantiene los faltantes en los predictores: se imputan despues de dividir
    los datos, dentro de cada fold de validacion cruzada.
    """
    missing_columns = sorted(set(COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas obligatorias: {missing_columns}")
    out = df.loc[:, COLUMNS].copy()
    n_original = len(out)
    null_before = {col: int(v) for col, v in out.isna().sum().items()}
    for col in [TARGET, *CATEGORICAL]:
        out[col] = out[col].astype("string").str.strip()
        out[col] = out[col].replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "?": pd.NA})
    for col in NUMERICAL:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    # Reglas de consistencia biologica; no introducir umbrales arbitrarios.
    invalidated = {}
    for col in NUMERICAL:
        invalid = out[col].notna() & (out[col] <= 0)
        invalidated[col] = int(invalid.sum())
        out.loc[invalid, col] = np.nan
    invalid_year = out["year"].notna() & (~out["year"].isin([2007, 2008, 2009]))
    invalidated["year"] = int(invalid_year.sum())
    out.loc[invalid_year, "year"] = np.nan
    # Solo clases conocidas; no rellenar etiqueta objetivo.
    invalid_target = out[TARGET].notna() & (~out[TARGET].isin(["Adelie", "Chinstrap", "Gentoo"]))
    invalidated[TARGET] = int(invalid_target.sum())
    out.loc[invalid_target, TARGET] = pd.NA
    target_missing = int(out[TARGET].isna().sum())
    out = out.dropna(subset=[TARGET])
    duplicates = int(out.duplicated().sum())
    out = out.drop_duplicates(ignore_index=True)
    out = out.sort_values(by=["species", "island", "year", "bill_length_mm", "bill_depth_mm"], kind="stable", na_position="last").reset_index(drop=True)
    audit = {
        "original_rows": n_original,
        "final_rows": int(len(out)),
        "duplicates_removed": duplicates,
        "rows_removed_missing_target": target_missing,
        "initial_missing_by_column": null_before,
        "remaining_missing_by_column": {c: int(v) for c, v in out.isna().sum().items()},
        "values_invalidated_by_column": invalidated,
        "classes": {str(k): int(v) for k, v in out[TARGET].value_counts().sort_index().items()},
        "note": "La imputacion numerica/categorica y el escalado se hacen en Pipeline por fold, no en el CSV limpio.",
    }
    return out, audit


def preparar_dataset(repo_root: Path) -> dict:
    repo_root = Path(repo_root).resolve()
    raw_path = repo_root / "datos/datos_ini/penguins_original.csv"
    clean_path = repo_root / "datos/datos_limp/penguins_limpios.csv"
    audit_path = repo_root / "datos/datos_limp/auditoria_preparacion.json"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_path)
    cleaned, audit = limpiar_datos(df)
    if len(cleaned) < 10 or cleaned[TARGET].nunique() < 2:
        raise ValueError("Datos insuficientes para clasificacion estratificada")
    cleaned.to_csv(clean_path, index=False, na_rep="", float_format="%.6g")
    audit["raw_sha256"] = sha256_file(raw_path)
    audit["clean_sha256"] = sha256_file(clean_path)
    audit["raw_path"] = raw_path.relative_to(repo_root).as_posix()
    audit["clean_path"] = clean_path.relative_to(repo_root).as_posix()
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    print(json.dumps(preparar_dataset(args.repo_root), ensure_ascii=False, indent=2))
