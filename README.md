# TelecomX-Analysis2: Análisis Predictivo de Evasión de Clientes

## Introducción

El objetivo del proyecto **TelecomX-Analysis2** es desarrollar un análisis predictivo para identificar los factores que influyen en la cancelación de clientes (Evasión) en la empresa TelecomX y proponer estrategias de retención. Se utilizó el conjunto de datos `TelecomX_Data_standardized.csv` (7267 muestras, 31 columnas tras preprocesamiento), con un desbalance de clases (74.28% Evasión=0, 25.72% Evasión=1). Se entrenaron y evaluaron dos modelos: **Regresión Logística** (con estandarización) y **Random Forest** (sin estandarización). Este proyecto incluye un análisis exploratorio de datos (EDA), preparación de datos, modelado, evaluación, interpretación de resultados, y estrategias de retención basadas en los factores clave identificados.

## Estructura del Proyecto

El repositorio está organizado en las siguientes carpetas y archivos:

- **data/**: Contiene el archivo de datos preprocesado.
  - `TelecomX_Data_standardized.csv`: Dataset con variables codificadas y estandarizadas.

- **notebooks/**: Contiene el cuaderno principal con el análisis completo.
  - `data_preparation.ipynb`: Notebook de Google Colab que abarca todas las etapas del análisis.

- **img/**: Contiene las visualizaciones generadas.
  - `correlation_matrix.png`: Matriz de correlación de variables.
  - `boxplots_evasion.png`: Boxplots de Antigüedad y Cargos_Totales por Evasión.
  - `scatter_evasion.png`: Scatter plot de Antigüedad vs. Cargos_Totales.
  - `confusion_matrix_lr.png`: Matriz de confusión para Regresión Logística.
  - `confusion_matrix_rf.png`: Matriz de confusión para Random Forest.
  - `feature_importance_lr.png`: Variables más relevantes para Regresión Logística.
  - `feature_importance_rf.png`: Variables más relevantes para Random Forest.

- **README.md**: Este archivo, que describe el proyecto y su ejecución.

## Proceso de Preparación de Datos

### Clasificación de Variables

El dataset original contiene:

- **Numéricas (4)**: Antigüedad (meses), Cargos_Mensuales, Cargos_Totales, Cargos_Diarios.
- **Categóricas (17, antes de codificación)**: Incluyen Género, Pareja, Dependientes, Servicio_Telefónico, Servicio_Internet, Contrato, Método_Pago, etc.
- **Variable objetivo**: Evasión (binaria: 0=Activo, 1=Canceló).

### Codificación y Normalización

- **Eliminación de columnas**: Se eliminó `ID_Cliente` por ser irrelevante para la predicción.
- **Codificación de categóricas**: Transformación mediante one-hot encoding, resultando en 31 columnas.
- **Estandarización**: Se aplicó a variables numéricas usando `StandardScaler` para Regresión Logística. Random Forest no la requiere.

### División de Datos

- Los datos se dividieron en entrenamiento (80%, 5813 muestras) y prueba (20%, 1454 muestras) usando `train_test_split` con `stratify=y`.
- La estratificación asegura proporciones de clases consistentes, esencial por el desbalance.

## Justificaciones para la Modelización

Se seleccionaron dos modelos:

### Regresión Logística

- **Justificación**: Modelo lineal, interpretable y adecuado para relaciones lineales.
- **Rendimiento**: 
  - Accuracy: 80.26%
  - Recall Evasión=1: 54.28%
  - F1-score: 0.59
- **Observaciones**: Estable, sin overfitting, pero posible underfitting.

### Random Forest

- **Justificación**: Modelo no lineal, robusto para relaciones complejas.
- **Rendimiento**:
  - Accuracy: 78.13%
  - Recall Evasión=1: 48.40%
  - F1-score: 0.53
- **Observaciones**: Presenta overfitting en entrenamiento (accuracy: 99.11%, recall: 97.99%).

### Consideraciones sobre el Desbalance

- El desbalance (25.72% Evasión=1) limita el recall.
- No se aplicó balanceo (ej. SMOTE) para preservar representatividad. Se sugiere como mejora.
- Se priorizó el **recall para Evasión=1**.

## Ejemplos de Gráficos e Insights del EDA

### Matriz de Correlación

**Insight**: Antigüedad tiene correlación negativa fuerte con Evasión. Cargos_Totales y Contrato_Month-to-month tienen correlaciones positivas.

### Boxplots de Antigüedad y Cargos Totales

**Insight**: Clientes que cancelan tienen menor antigüedad (~10 meses) y mayores cargos totales.

### Scatter Plot: Antigüedad vs. Cargos Totales

**Insight**: Clientes que cancelan se concentran en baja antigüedad (<20 meses) y cargos moderados a altos.

### Matrices de Confusión

- **Regresión Logística**: Mejor recall para Evasión=1 (54.28%).
- **Random Forest**: Recall más bajo (48.40%).

### Variables Relevantes

**Insight**: Antigüedad, Cargos_Totales y Contrato_Month-to-month son factores clave. Contratos a largo plazo reducen cancelación.

## Instrucciones para Ejecutar el Cuaderno

### Requisitos

- **Entorno**: Google Colab (recomendado) o local con Python 3.7+.
- **Librerías necesarias**:

```bash
pip install pandas numpy seaborn matplotlib plotly scikit-learn
```

En Colab, suelen estar preinstaladas.

### Pasos para Ejecutar

1. **Clonar el repositorio** (si usas local):

```bash
git clone https://github.com/carlo55anchez/TelecomX-Analysis2.git
cd TelecomX-Analysis2
```

2. **Abrir el cuaderno**:

- En Colab: Sube `notebooks/data_preparation.ipynb` o ábrelo desde el repositorio.
- En local: Usa Jupyter o VS Code:

```bash
jupyter notebook notebooks/data_preparation.ipynb
```

3. **Cargar los datos**:

```python
url = 'https://raw.githubusercontent.com/carlo55anchez/TelecomX-Analysis2/main/data/TelecomX_Data_standardized.csv'
df = pd.read_csv(url)
```

- Si estás en local, asegúrate de que el archivo esté en `data/`.

4. **Ejecutar las celdas**:

- Ejecuta en orden (`Shift + Enter`).
- Las visualizaciones se guardan en `img/`.

5. **Descargar gráficos (opcional, Colab)**:

```python
import shutil
from google.colab import files
shutil.make_archive('img_archive', 'zip', 'img')
files.download('img_archive.zip')
```

### Notas

- Verifica conexión a internet para cargar el dataset desde la URL.
- Si hay errores, revisa versiones de librerías o mensajes en Colab.

## Conclusión

El proyecto **TelecomX-Analysis2** identifica factores clave en la cancelación de clientes como Antigüedad, Cargos_Totales y tipo de contrato. Propone estrategias de retención basadas en estos factores. El cuaderno `data_preparation.ipynb` y las visualizaciones en `img/` constituyen una base sólida para análisis futuros. Se recomiendan mejoras como balanceo de clases y ajuste de hiperparámetros para mejorar el rendimiento del modelo.
