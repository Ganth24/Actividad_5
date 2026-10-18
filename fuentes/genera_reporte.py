"""Genera PDF de evaluacion desde resultados locales realmente producidos."""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
    NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / 'resultados/resultados_experimento.json').read_text(encoding='utf-8'))
AUDIT = json.loads((ROOT / 'datos/datos_limp/auditoria_preparacion.json').read_text(encoding='utf-8'))
PDF = ROOT / 'reportes/reporte_tecnico.pdf'
PDF.parent.mkdir(parents=True, exist_ok=True)

NAVY=colors.HexColor('#172B49')
BLUE=colors.HexColor('#235C88')
TEAL=colors.HexColor('#047E86')
MUTED=colors.HexColor('#526378')
GRAY=colors.HexColor('#EEF2F6')
LINE=colors.HexColor('#DBE3EB')
RED=colors.HexColor('#934030')

fontbase='/usr/share/fonts/truetype/dejavu'
pdfmetrics.registerFont(TTFont('DejaVu',f'{fontbase}/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVu-Bold',f'{fontbase}/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold')

styles={
    'kicker':ParagraphStyle('kicker',fontName='DejaVu-Bold',fontSize=10,leading=15,textColor=TEAL,spaceAfter=14),
    'cover':ParagraphStyle('cover',fontName='DejaVu-Bold',fontSize=27,leading=35,textColor=NAVY,spaceAfter=14),
    'subtitle':ParagraphStyle('subtitle',fontName='DejaVu',fontSize=12,leading=19,textColor=MUTED,spaceAfter=20),
    'h1':ParagraphStyle('h1',fontName='DejaVu-Bold',fontSize=14.5,leading=21,textColor=NAVY,spaceBefore=18,spaceAfter=10,keepWithNext=True),
    'h2':ParagraphStyle('h2',fontName='DejaVu-Bold',fontSize=10.4,leading=16,textColor=BLUE,spaceBefore=11,spaceAfter=7,keepWithNext=True),
    'body':ParagraphStyle('body',fontName='DejaVu',fontSize=8.8,leading=14.4,textColor=NAVY,spaceAfter=9),
    'small':ParagraphStyle('small',fontName='DejaVu',fontSize=7.8,leading=12.4,textColor=MUTED,spaceAfter=6),
    'caption':ParagraphStyle('caption',fontName='DejaVu',fontSize=7.6,leading=11.9,textColor=MUTED,spaceAfter=12,spaceBefore=5),
    'tablehead':ParagraphStyle('tablehead',fontName='DejaVu-Bold',fontSize=7.2,leading=11,textColor=colors.white),
    'table':ParagraphStyle('table',fontName='DejaVu',fontSize=7.15,leading=11.3,textColor=NAVY),
    'bullet':ParagraphStyle('bullet',fontName='DejaVu',fontSize=8.7,leading=14.4,textColor=NAVY,bulletIndent=12,leftIndent=17,spaceAfter=5),
    'status':ParagraphStyle('status',fontName='DejaVu-Bold',fontSize=8.5,leading=14,textColor=RED),
    'kpi':ParagraphStyle('kpi',fontName='DejaVu-Bold',fontSize=20,leading=26,textColor=TEAL,alignment=TA_CENTER),
    'kpilabel':ParagraphStyle('kpilabel',fontName='DejaVu',fontSize=8,leading=12,textColor=MUTED,alignment=TA_CENTER),
}

def P(s,style='body'):return Paragraph(s,styles[style])
def txt(s):return escape(str(s))
def B(s):return f'<b>{txt(s)}</b>'
def section(num,name):return P(f'{num}. {name}','h1')
def sub(name):return P(name,'h2')
def bullet(text):return P('&#8226; '+text,'bullet')
def tab(headers, data, widths, colorfirst=False):
    rows=[[P(txt(s),'tablehead') for s in headers]]
    rows += [[P(str(s),'table') for s in row] for row in data]
    tbl=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT',splitByRow=1)
    commands=[('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LINEBELOW',(0,0),(-1,0),0.8,BLUE),('LINEBELOW',(0,1),(-1,-1),0.35,LINE),]
    for i in range(2,len(rows),2):commands.append(('BACKGROUND',(0,i),(-1,i),GRAY))
    tbl.setStyle(TableStyle(commands))
    return tbl

def img(name, width=488):
    p=ROOT/'figuras'/name
    from PIL import Image as PILImage
    iw,ih=PILImage.open(p).size
    return Image(str(p),width=width,height=width*ih/iw,hAlign='CENTER')

def draw_page(canvas,doc):
    canvas.saveState()
    w,h=A4
    canvas.setStrokeColor(LINE)
    canvas.line(48,h-44,w-48,h-44)
    canvas.setFillColor(TEAL)
    canvas.setFont('DejaVu-Bold',8)
    canvas.drawString(48,h-34,'ACTIVIDAD 5  /  MODELOS Y MLOPS')
    canvas.setFillColor(MUTED)
    canvas.setFont('DejaVu',7.4)
    canvas.drawRightString(w-48,33,f'18 OCT 2026  •  PAG. {doc.page}')
    canvas.restoreState()

class Doc(BaseDocTemplate):pass

doc=Doc(str(PDF),pagesize=A4,rightMargin=49,leftMargin=49,topMargin=56,bottomMargin=52,
        title='Actividad 5 - Reporte tecnico Palmer Penguins',author='Proyecto aplicado de aprendizaje automatico')
frame=Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,id='main',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
doc.addPageTemplates(PageTemplate(id='main',frames=frame,onPage=draw_page))
story=[]
def add(*components):story.extend(components)

def fmt(v):return f'{v:.4f}'
rl=RESULTS['models']['regresion_logistica']
rf=RESULTS['models']['random_forest']
st=RESULTS['statistical_comparison']

# PORTADA
add(Spacer(1,48),P('REPORTE TÉCNICO  •  PROYECTO APLICADO','kicker'),
    P('Clasificación multiclase de pingüinos: datos, modelos y reproducibilidad','cover'),
    P('Actividad 5  |  Palmer Penguins  |  Regresión logística vs. Random Forest','subtitle'),
    Spacer(1,21))
kpis=Table([[
    [P('344','kpi'),P('Registros originales','kpilabel')],
    [P('22','kpi'),P('Configuraciones evaluadas','kpilabel')],
    [P('5','kpi'),P('Folds estratificados','kpilabel')],
    [P('2','kpi'),P('Modelos comparados','kpilabel')]]],colWidths=[124,124,124,124])
kpis.setStyle(TableStyle([('BOX',(0,0),(-1,-1),.6,LINE),('INNERGRID',(0,0),(-1,-1),.4,LINE),
    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),13),('BOTTOMPADDING',(0,0),(-1,-1),13)]))
add(kpis,Spacer(1,32),sub('Propósito del proyecto'),
    P('Preparar, auditar y versionar un conjunto de datos público; entrenar y ajustar dos algoritmos con validación cruzada; evaluar sobre un test reservado; documentar resultados, incertidumbre y deuda técnica; e implementar el registro de experimentos mediante MLflow.'),
    Spacer(1,15),P('ESTADO DE VALIDACIÓN','status'),
    P('Se ejecutaron los pipelines, Grid Search, pruebas unitarias y gráficas en un entorno offline. <b>El registro MLflow no fue ejecutado aquí</b> por no disponer de la librería ni acceso para instalarla. El repositorio Git está preparado localmente, pero no ha sido publicado en una cuenta de GitHub. No se incluyen capturas ficticias.'),
    Spacer(1,28),P('Versión 1.0  ·  18 de octubre de 2026  ·  Proyecto reproducible con archivos adjuntos','small'),
    PageBreak())

add(section('1','Resumen ejecutivo y hallazgos'),
    P('El proyecto utiliza 344 registros de las especies Adelie, Chinstrap y Gentoo. Los datos se mantuvieron completos en número de filas tras su depuración; los valores faltantes de variables predictoras se resuelven <b>exclusivamente dentro del pipeline</b> para prevenir fuga de información.'),
    P(f'La regresión logística consiguió F1 macro de <b>{fmt(rl["test_metrics"]["f1_macro"])}</b> en 69 observaciones de test, frente a <b>{fmt(rf["test_metrics"]["f1_macro"])}</b> de Random Forest. En validación cruzada, las medias de F1 macro fueron {fmt(rl["cv_f1_mean"])} y {fmt(rf["cv_f1_mean"])}, respectivamente. Las diferencias favorecen a la regresión logística en este conjunto, aunque el tamaño y origen de la muestra limitan la extrapolación.'),
    sub('Diseño para evitar sobreajuste y filtración'),
    bullet('<b>Separación:</b> 275 registros en entrenamiento y 69 en test, estratificados por especie y con semilla 42.'),
    bullet('<b>Selección:</b> GridSearchCV con 5 pliegues estratificados dentro de entrenamiento; el test no interviene en la búsqueda.'),
    bullet('<b>Transformación:</b> imputación y codificación dentro de ColumnTransformer/Pipeline; normalización numérica solo para regresión logística.'),
    bullet('<b>Comparación:</b> métricas homogéneas y bootstrap pareado con 2,000 remuestreos de los mismos casos de prueba.'),
    section('2','Datos: origen, variables y supuestos'),
    P('El conjunto público <b>Palmer Penguins</b> proviene del proyecto palmerpenguins, desarrollado para facilitar el aprendizaje de análisis de datos con observaciones de pingüinos en el entorno de Palmer Station, Antártida (años 2007 a 2009). La copia CSV utilizada está incluida en el ZIP y procede de la distribución local de plotnine; su estructura coincide con el archivo público documentado en GitHub.'),
    sub('Diccionario de datos'),
    tab(['Campo','Tipo','Uso y definición'],[
      ('species','Categórica','Variable objetivo: Adelie, Chinstrap o Gentoo.'),
      ('island','Categórica','Isla de muestreo (Biscoe, Dream, Torgersen).'),
      ('bill_length_mm','Numérica','Longitud del pico (mm).'),
      ('bill_depth_mm','Numérica','Profundidad del pico (mm).'),
      ('flipper_length_mm','Numérica','Longitud de aleta (mm).'),
      ('body_mass_g','Numérica','Masa corporal (g).'),
      ('sex','Categórica','Sexo registrado o nulo.'),
      ('year','Entero','Año del registro (se conserva, no se usa como predictor).'),
     ],[130,80,286]),
    P('Supuestos: registros individuales de observación; mediciones corporales cuantitativas en sus unidades originales; no se asume representatividad fuera de las islas y años muestreados. `year` se excluye como predictor por su posible efecto espurio de campaña; `island` se conserva, pero se reconoce el riesgo de dependencia geográfica.','small'))

add(section('3','Preparación, calidad y versionamiento'),
    P('El archivo original se conserva sin edición. El script <b>fuentes/datos_prep.py</b> comprueba el esquema, limpia cadenas y tokens de faltantes, convierte numéricos, marca mediciones no positivas como nulas, valida el rango 2007-2009, elimina filas sin etiqueta y retira duplicados exactos. El resultado se guarda en CSV con un archivo de auditoría JSON que contiene conteos y huellas SHA-256.'),
    sub('Resultado de auditoría de calidad'),
    tab(['Control','Observación en la copia original'],[
      ('Registros antes/después','344 / 344'),
      ('Duplicados exactos eliminados','0 (la regla se implementa y prueba)'),
      ('Etiquetas objetivo sin valor','0'),
      ('Ausentes en medidas numéricas','2 por cada variable numérica'),
      ('Ausentes en sexo','11'),
      ('Ausentes en isla / año / especie','0 / 0 / 0'),
      ('Invalidados por valor físicamente imposible','0 en las medidas numéricas originales'),
    ],[232,264]),
    sub('Limpieza vs. transformación estadística'),
    P('Se mantienen los nulos de los predictores en <b>datos_limp/penguins_limpios.csv</b> a propósito. La mediana de cada columna numérica, la moda de cada variable categórica y las escalas se aprenden <b>solo del fold de entrenamiento</b> correspondiente. Las variables `island` y `sex` se codifican con OneHotEncoder; las categorías nuevas se ignoran sin error. Este orden impide que estadísticas del test influyan en entrenamiento.'),
    sub('Estrategia de trazabilidad'),
    P(f'SHA-256 original: <font size="7">{AUDIT["raw_sha256"]}</font>'),
    P(f'SHA-256 limpio: <font size="7">{AUDIT["clean_sha256"]}</font>'),
    P('Se usan Git y CHANGELOG.md para registrar versiones y decisiones. La auditoría enlaza los dos ficheros mediante sus hashes; en una corrida MLflow real el script registra ambos SHA-256 como etiquetas. El ZIP incluye historial Git local; su publicación remota en GitHub está pendiente de permisos.'),
    section('4','Exploración: patrones, faltantes y anomalías'),
    img('01_eda_especies_pico.png',487),
    P('<b>Figura 1.</b> Adelie (152 registros), Gentoo (124) y Chinstrap (68): existe desbalance moderado. La longitud y profundidad del pico muestran regiones separables, aunque solapamientos entre especies; favorece comparar un límite de decisión lineal con un conjunto de árboles.','caption'),
    img('02_nulos_y_masa.png',487),
    P('<b>Figura 2.</b> Sexo contiene 11 nulos y cada medida física, 2; por ello es pertinente una imputación aprendida en cada fold. La masa corporal difiere entre especies; se evitan reglas de eliminación de valores atípicos sin evidencia biológica. El criterio exploratorio 1.5 × IQR no marcó valores extremos por ninguna medida.','caption'))

add(section('5','Modelos, hiperparámetros y evaluación'),
    P('La regresión logística se usa como línea base interpretable y regularizada; se estandarizan sus variables numéricas para evitar dependencia de la escala. Random Forest modela interacciones y relaciones no lineales y no necesita reescalado numérico. Ambos comparten tratamiento de nulos/categorías y la misma partición, semilla y pliegues.'),
    tab(['Modelo','Hiperparámetros de Grid Search','Justificación'],[
      ('Regresión logística','C = 0.1, 1, 10; class_weight = None, balanced (6 candidatos).','Regularización y compensación potencial del desbalance.'),
      ('Random Forest','n_estimators = 100, 200; max_depth = None, 6; min_samples_leaf = 1, 3; class_weight = None, balanced (16 candidatos).','Estabilidad del ensemble, control de complejidad y peso de clases.'),
    ],[112,216,168]),
    P('El criterio de selección es <b>F1 macro</b> (promedio no ponderado de F1 por especie), que da visibilidad a la clase con menos registros. En test se calculan accuracy, precision macro, recall macro, F1 macro y ROC-AUC multiclase One-vs-Rest (macro). Para ROC-AUC se usan las probabilidades del modelo, no las etiquetas duras.'),
    sub('Hiperparámetros seleccionados por CV'),
    tab(['Modelo','Configuración ganadora'],[
      ('Regresión logística','C = 10; class_weight = None; max_iter = 3000.'),
      ('Random Forest','n_estimators = 100; max_depth = None; min_samples_leaf = 1; class_weight = None.'),
    ],[139,357]),
    section('6','Resultados cuantitativos'),
    tab(['Métrica','Regresión logística','Random Forest'],[
      ('F1 macro CV, media ± desv. est.',f'{rl["cv_f1_mean"]:.4f} ± {rl["cv_f1_std"]:.4f}',f'{rf["cv_f1_mean"]:.4f} ± {rf["cv_f1_std"]:.4f}'),
      ('Accuracy test',fmt(rl['test_metrics']['accuracy']),fmt(rf['test_metrics']['accuracy'])),
      ('Precision macro test',fmt(rl['test_metrics']['precision_macro']),fmt(rf['test_metrics']['precision_macro'])),
      ('Recall macro test',fmt(rl['test_metrics']['recall_macro']),fmt(rf['test_metrics']['recall_macro'])),
      ('F1 macro test',fmt(rl['test_metrics']['f1_macro']),fmt(rf['test_metrics']['f1_macro'])),
      ('ROC-AUC OvR macro test',fmt(rl['test_metrics']['roc_auc_ovr_macro']),fmt(rf['test_metrics']['roc_auc_ovr_macro'])),
    ],[213,141,142]),
    P('Valores calculados sobre 69 registros de prueba independientes del Grid Search. Las desviaciones CV describen dispersión entre folds; no son intervalos de confianza ni tests de hipótesis.','small'),
    img('05_comparacion_modelos.png',487),
    P('<b>Figura 3.</b> Comparación en las mismas instancias de test: la regresión logística supera ligeramente al bosque en F1 macro. ROC-AUC = 1.000 en ambos modelos expresa separación probabilística observada, no identidad de predicciones por clase.','caption'))

add(section('7','Interpretación visual y análisis estadístico'),
    sub('Errores sobre test independiente'),
    Table([[img('03_confusion_regresion_logistica.png',243),img('03_confusion_random_forest.png',243)]],
        colWidths=[248,248],style=TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0)])),
    P('<b>Figura 4.</b> Izquierda: regresión logística; derecha: Random Forest. La matriz de confusión hace visible que un score agregado casi perfecto puede ocultar errores aislados, especialmente relevantes cuando las clases son de diferente tamaño.','caption'),
    Table([[img('04_roc_regresion_logistica.png',243),img('04_roc_random_forest.png',243)]],
        colWidths=[248,248],style=TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0)])),
    P('<b>Figura 5.</b> Curvas ROC One-vs-Rest para las tres especies. No se pueden inferir probabilidades calibradas solo a partir de esta métrica; calibración y validación fuera de muestra quedan pendientes.','caption'),
    sub('Bootstrap pareado sobre F1 macro'),
    P(f'Diferencia definida como <b>F1(Random Forest) - F1(regresión logística)</b>: <b>{st["point_difference"]:+.4f}</b>. Con {st["resamples"]:,} remuestreos estratificados de los 69 casos de test (mismos índices para ambos modelos), el intervalo percentil 95 % fue <b>[{st["ci95_percentile"][0]:.4f}, {st["ci95_percentile"][1]:.4f}]</b>. La diferencia observada es pequeña y el extremo superior incluye 0; no se afirma superioridad estadística generalizable.'),
    P('Limitaciones metodológicas: conjunto pequeño, observaciones del mismo ecosistema y distribuciones de isla/especie potencialmente correlacionadas. Para comparar rendimiento real fuera de distribución se necesitarían conjuntos externos o particiones por campañas geográficas/temporales, además de repetición con distintas semillas.','small'))

add(section('8','Registro en MLflow: implementación y estado real'),
    P('<b>Implementación disponible:</b> `fuentes/train.py` define un experimento local con nombre explícito, una corrida final por modelo, 6 + 16 corridas hijas para cada candidato de Grid Search y una corrida comparativa. Registra hiperparámetros, métricas CV/test, huellas de los CSV, gráficos, audit JSON y el pipeline completo (preprocesado y estimador) con firma y ejemplo.'),
    P('<b>Validación efectivamente realizada:</b> el modo `--sin-mlflow` verificó el entrenamiento y los artefactos en disco. <b>MLflow no se ejecutó</b> porque no estaba instalado y no había conectividad disponible para obtenerlo. No existen capturas auténticas del panel ni identificadores de corridas MLflow correspondientes a esta ejecución. El campo `tracking_mlflow_ejecutado` del JSON es `false`.'),
    sub('Procedimiento de evidencia para completar al reproducir'),
    bullet('Instalar `requirements.txt`, ejecutar `python fuentes/train.py` sin la bandera offline y abrir la UI mediante `mlflow ui --backend-store-uri ./mlruns --port 5000`.'),
    bullet('Capturar la lista del experimento con métricas, el detalle de parámetros/métricas, y la sección de artefactos/modelo. Guardar las imágenes reales bajo `evidencias_mlflow/capturas/`.'),
    bullet('Verificar hashes de origen y limpieza en las etiquetas de cada run; conservar el historial Git y, cuando haga falta, exportar el FileStore de tracking a almacenamiento persistente.'),
    P('Esta distinción entre <b>código implementado</b> y <b>evidencia verificada</b> es indispensable para no sobreestimar la trazabilidad entregada.'),
    section('9','Deuda técnica y plan de mejora MLOps'),
    tab(['Riesgo identificado','Impacto','Acción recomendada'],[
      ('Dependencias con rangos amplios','Variación en futuras instalaciones.','Publicar lockfile/imagen Docker con versiones exactas y verificación CI.'),
      ('Tracking local de archivos','Pérdida de corridas al borrar sesión.','Backend y almacén de artefactos persistentes con controles de acceso.'),
      ('Muestra y dominio limitados','Optimismo, drift geográfico o temporal.','Evaluación externa por isla/año y vigilancia de datos y desempeño.'),
      ('Ausencia de aprobación para despliegue','Riesgo de modelo sin revisión.','Model Registry, etapas, auditoría, criterios de promoción y rollback.'),
      ('Calidad de entrada variable','Errores por nuevas categorías o sensores.','Checks de esquema/rangos, tests de inferencia, alertas y monitoreo.'),
    ],[138,145,213]),
    section('10','Conclusiones y referencias'),
    P('La preparación reproducible, la validación estratificada y el holdout sustentan una comparación consistente de dos enfoques. Para este dataset, la regresión logística es la opción preferible como línea base por su simplicidad y resultado observado; la diferencia es pequeña y no respalda una afirmación universal. La siguiente etapa necesaria para cierre completo de la rúbrica es ejecutar el tracking real y publicar el repositorio en una cuenta con autorización.'),
    P('<b>Referencias:</b> Horst et al., <i>palmerpenguins</i> (https://github.com/allisonhorst/palmerpenguins); scikit-learn, <i>GridSearchCV</i> (https://scikit-learn.org); MLflow, <i>Tracking</i> (https://mlflow.org/docs/latest/ml/tracking/).','small'))

doc.build(story)
print(PDF)
