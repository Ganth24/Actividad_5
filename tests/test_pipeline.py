import numpy as np
import pandas as pd
from fuentes.datos_prep import limpiar_datos
from fuentes.train import pipeline_modelo


def test_limpieza_rechaza_duplicados_y_valores_imposibles():
    row = dict(species="Adelie", island="Torgersen", bill_length_mm=39.1,
        bill_depth_mm=18.7, flipper_length_mm=181, body_mass_g=3750,
        sex="male", year=2007)
    another = dict(row, species="Gentoo", body_mass_g=-1, sex="NA")
    df = pd.DataFrame([row,row,another])
    clean, audit = limpiar_datos(df)
    assert len(clean) == 2
    assert audit["duplicates_removed"] == 1
    assert audit["values_invalidated_by_column"]["body_mass_g"] == 1
    assert clean["body_mass_g"].isna().sum() == 1
    assert clean["sex"].isna().sum() == 1


def test_preprocesamiento_no_falla_ante_nulos_ni_categorias_nuevas():
    X = pd.DataFrame({"bill_length_mm":[39,45,48,36,46,49],
         "bill_depth_mm":[19,17,14,18,16,13],
         "flipper_length_mm":[185,190,210,184,193,215],
         "body_mass_g":[3700,4000,np.nan,3500,4100,5000],
         "island":["Dream","Biscoe","Torgersen","Dream","Biscoe","Torgersen"],
         "sex":["male","female",None,"male","female","male"]})
    y = ["Adelie","Chinstrap","Gentoo","Adelie","Chinstrap","Gentoo"]
    for name in ["regresion_logistica", "random_forest"]:
        model = pipeline_modelo(name).fit(X,y)
        xnew = X.iloc[:1].copy()
        xnew.loc[xnew.index[0],"island"] = "Nueva_isla"
        assert len(model.predict(xnew)) == 1
