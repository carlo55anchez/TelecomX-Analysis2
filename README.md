TelecomX-Analysis2: Análisis Predictivo de Evasión de Clientes

Propósito del Análisis

El proyecto TelecomX-Analysis2 tiene como objetivo principal predecir la cancelación de clientes (churn) en la empresa TelecomX, identificando los factores clave que influyen en la decisión de los clientes de cancelar sus servicios (Evasión). Utilizando técnicas de aprendizaje automático, se desarrollaron modelos predictivos (Regresión Logística y Random Forest) para clasificar clientes como activos (Evasión=0) o cancelados (Evasión=1), basándose en variables demográficas, de servicios y financieras. El análisis también proporciona estrategias de retención basadas en los factores identificados, como Antigüedad, Cargos_Totales, y Contrato_Month-to-month, para reducir la tasa de churn y mejorar la lealtad de los clientes.

El dataset utilizado contiene 7267 muestras con 31 columnas (tras preprocesamiento), con un desbalance de clases (74.28% Evasión=0, 25.72% Evasión=1). Este proyecto abarca un análisis exploratorio de datos (EDA), preparación de datos, modelado, evaluación, e interpretación de resultados, ofreciendo una base sólida para decisiones estratégicas.

Estructura del Proyecto

El repositorio está organizado en las siguientes carpetas y archivos:





data/: Contiene el archivo de datos preprocesado.





TelecomX_Data_standardized.csv: Dataset con variables codificadas y estandarizadas, listo para análisis.



notebooks/: Contiene el cuaderno principal con el análisis completo.





data_preparation.ipynb: Notebook de Google Colab que incluye todas las etapas del análisis (preparación, modelado, evaluación, e informe).



img/: Contiene visualizaciones generadas durante el análisis.





correlation_matrix.png: Matriz de correlación de variables.



boxplots_evasion.png: Boxplots de Antigüedad y Cargos_Totales por Evasión.



scatter_evasion.png: Scatter plot de Antigüedad vs. Cargos_Totales.



confusion_matrix_lr.png: Matriz de confusión para Regresión Logística.



confusion_matrix_rf.png: Matriz de confusión para Random Forest.



feature_importance_lr.png: Variables más relevantes para Regresión Logística.



feature_importance_rf.png: Variables más relevantes para Random Forest.



README.md: Este archivo, que describe el proyecto y su ejecución.

Proceso de Preparación de Datos

La preparación de datos se realizó en el cuaderno data_preparation.ipynb y consta de las siguientes etapas:

Clasificación de Variables

El dataset original contiene variables categóricas y numéricas:





Numéricas (4): Antigüedad (meses), Cargos_Mensuales, Cargos_Totales, Cargos_Diarios.



Categóricas (17, antes de codificación): Incluyen Género, Pareja, Dependientes, Servicio_Telefónico, Servicio_Internet, Contrato, Método_Pago, etc.



Variable objetivo: Evasión (binaria: 0=Activo, 1=Canceló).

Codificación y Normalización





Eliminación de columnas: Se eliminó ID_Cliente por ser irrelevante para la predicción.



Codificación de categóricas: Las variables categóricas se codificaron usando one-hot encoding, generando 31 columnas en el DataFrame df_encoded.



Estandarización: Las variables numéricas (Antigüedad, Cargos_Mensuales, Cargos_Totales, Cargos_Diarios) se estandarizaron (media=0, desviación estándar=1) solo para Regresión Logística, usando StandardScaler, ya que este modelo es sensible a la escala. Random Forest no requirió estandarización, ya que usa particiones basadas en umbrales.

División de Datos





Los datos se dividieron en conjuntos de entrenamiento (80%, 5813 muestras) y prueba (20%, 1454 muestras) usando train_test_split con stratify=y para mantener el desbalance (74.28% Evasión=0, 25.72% Evasión=1).



La estratificación asegura que las proporciones de clases sean consistentes en ambos conjuntos, crucial debido al desbalance.

Justificaciones para la Modelización

Se entrenaron dos modelos: Regresión Logística y Random Forest, seleccionados por las siguientes razones:





Regresión Logística:





Justificación: Modelo simple e interpretable, adecuado para relaciones lineales. Requiere estandarización para evitar sesgos por las magnitudes dispares de las variables (por ejemplo, Cargos_Totales: 18.8-8684.8).



Rendimiento: Accuracy 80.26%, recall para Evasión=1 54.28%. Estable (sin overfitting), pero con posible underfitting debido a su simplicidad.



Random Forest:





Justificación: Modelo robusto, capaz de capturar relaciones no lineales y manejar desbalance moderado. No requiere estandarización, ya que usa particiones basadas en umbrales.



Rendimiento: Accuracy 78.13%, recall para Evasión=1 48.40%. Muestra overfitting (accuracy entrenamiento: 99.11%, recall: 97.99%), debido a árboles profundos sin restricciones.



Desbalance: El desbalance (25.72% Evasión=1) limita el recall de ambos modelos. No se aplicó balanceo (por ejemplo, SMOTE) para mantener la representatividad inicial, pero se sugiere como mejora.



Métricas: Se priorizó el recall para Evasión=1 debido a la importancia de detectar clientes en riesgo de cancelación.

Ejemplos de Gráficos e Insights del EDA

El análisis exploratorio (Etapas 5-6) proporcionó los siguientes insights, ilustrados con gráficos:





Matriz de Correlación:





Gráfico:





Insight: Variables como Antigüedad (correlación negativa con `Evasión %

System: ¡Gracias por tu solicitud! Aquí tienes el contenido completo del README.md en formato Markdown, listo para que lo copies y pegues en el archivo README.md que ya generaste en tu repositorio https://github.com/carlo55anchez/TelecomX-Analysis2. Este contenido incluye todas las secciones solicitadas: propósito del análisis, estructura del proyecto, proceso de preparación de datos, justificaciones para la modelización, ejemplos de gráficos e insights del análisis exploratorio (EDA), e instrucciones para ejecutar el cuaderno. Las rutas de los gráficos utilizan el formato relativo img/nombre_del_archivo.png, asumiendo que los archivos PNG (correlation_matrix.png, boxplots_evasion.png, etc.) están en la carpeta img/ del repositorio. El contenido está envuelto en un <xaiArtifact> con el mismo artifact_id usado previamente para el README.md (4a9665e3-bd80-4ff4-b639-4b5190c1faf9).

 # TelecomX-Analysis2: Análisis Predictivo de Evasión de Clientes

Propósito del Análisis

El proyecto TelecomX-Analysis2 tiene como objetivo principal predecir la cancelación de clientes (churn) en la empresa TelecomX, identificando los factores clave que influyen en la decisión de los clientes de cancelar sus servicios (Evasión). Utilizando técnicas de aprendizaje automático, se desarrollaron modelos predictivos (Regresión Logística y Random Forest) para clasificar clientes como activos (Evasión=0) o cancelados (Evasión=1), basándose en variables demográficas, de servicios y financieras. El análisis también proporciona estrategias de retención basadas en los factores identificados, como Antigüedad, Cargos_Totales, y Contrato_Month-to-month, para reducir la tasa de churn y mejorar la lealtad de los clientes.

El dataset utilizado contiene 7267 muestras con 31 columnas (tras preprocesamiento), con un desbalance de clases (74.28% Evasión=0, 25.72% Evasión=1). Este proyecto abarca un análisis exploratorio de datos (EDA), preparación de datos, modelado, evaluación, e interpretación de resultados, ofreciendo una base sólida para decisiones estratégicas.

Estructura del Proyecto

El repositorio está organizado en las siguientes carpetas y archivos:





data/: Contiene el archivo de datos preprocesado.





TelecomX_Data_standardized.csv: Dataset con variables codificadas y estandarizadas, listo para análisis.



notebooks/: Contiene el cuaderno principal con el análisis completo.





data_preparation.ipynb: Notebook de Google Colab que incluye todas las etapas del análisis (preparación, modelado, evaluación, e informe).



img/: Contiene visualizaciones generadas durante el análisis.





correlation_matrix.png: Matriz de correlación de variables.



boxplots_evasion.png: Boxplots de Antigüedad y Cargos_Totales por Evasión.



scatter_evasion.png: Scatter plot de Antigüedad vs. Cargos_Totales.



confusion_matrix_lr.png: Matriz de confusión para Regresión Logística.



confusion_matrix_rf.png: Matriz de confusión para Random Forest.



feature_importance_lr.png: Variables más relevantes para Regresión Logística.



feature_importance_rf.png: Variables más relevantes para Random Forest.



README.md: Este archivo, que describe el proyecto y su ejecución.

Proceso de Preparación de Datos

La preparación de datos se realizó en el cuaderno data_preparation.ipynb y consta de las siguientes etapas:

Clasificación de Variables

El dataset original contiene variables categóricas y numéricas:





Numéricas (4): Antigüedad (meses), Cargos_Mensuales, Cargos_Totales, Cargos_Diarios.



Categóricas (17, antes de codificación): Incluyen Género, Pareja, Dependientes, Servicio_Telefónico, Servicio_Internet, Contrato, Método_Pago, etc.



Variable objetivo: Evasión (binaria: 0=Activo, 1=Canceló).

Codificación y Normalización





Eliminación de columnas: Se eliminó ID_Cliente por ser irrelevante para la predicción.



Codificación de categóricas: Las variables categóricas se codificaron usando one-hot encoding, generando 31 columnas en el DataFrame df_encoded.



Estandarización: Las variables numéricas (Antigüedad, Cargos_Mensuales, Cargos_Totales, Cargos_Diarios) se estandarizaron (media=0, desviación estándar=1) solo para Regresión Logística, usando StandardScaler, ya que este modelo es sensible a la escala. Random Forest no requirió estandarización, ya que usa particiones basadas en umbrales.

División de Datos





Los datos se dividieron en conjuntos de entrenamiento (80%, 5813 muestras) y prueba (20%, 1454 muestras) usando train_test_split con stratify=y para mantener el desbalance (74.28% Evasión=0, 25.72% Evasión=1).



La estratificación asegura que las proporciones de clases sean consistentes en ambos conjuntos, crucial debido al desbalance.

Justificaciones para la Modelización

Se entrenaron dos modelos: Regresión Logística y Random Forest, seleccionados por las siguientes razones:





Regresión Logística:





Justificación: Modelo simple e interpretable, adecuado para relaciones lineales. Requiere estandarización para evitar sesgos por las magnitudes dispares de las variables (por ejemplo, Cargos_Totales: 18.8-8684.8).



Rendimiento: Accuracy 80.26%, recall para Evasión=1 54.28%. Estable (sin overfitting), pero con posible underfitting debido a su simplicidad.



Random Forest:





Justificación: Modelo robusto, capaz de capturar relaciones no lineales y manejar desbalance moderado. No requiere estandarización, ya que usa particiones basadas en umbrales.



Rendimiento: Accuracy 78.13%, recall para Evasión=1 48.40%. Muestra overfitting (accuracy entrenamiento: 99.11%, recall: 97.99%), debido a árboles profundos sin restricciones.



Desbalance: El desbalance (25.72% Evasión=1) limita el recall de ambos modelos. No se aplicó balanceo (por ejemplo, SMOTE) para mantener la representatividad inicial, pero se sugiere como mejora.



Métricas: Se priorizó el recall para Evasión=1 debido a la importancia de detectar clientes en riesgo de cancelación.

Ejemplos de Gráficos e Insights del EDA

El análisis exploratorio (Etapas 5-6) proporcionó los siguientes insights, ilustrados con gráficos:





Matriz de Correlación:





Gráfico:





Insight: Variables como Antigüedad (correlación negativa con Evasión), Cargos_Totales (positiva), y Contrato_Month-to-month (positiva) mostraron relaciones significativas con la cancelación.



Boxplots de Antigüedad y Cargos Totales:





Gráfico:





Insight: Los clientes que cancelan (Evasión=1) tienen menor antigüedad (mediana ~10 meses) y mayores cargos totales, indicando que clientes recientes y con costos altos son más propensos a cancelar.



Scatter Plot: Antigüedad vs. Cargos Totales:





Gráfico:





Insight: Los clientes que cancelan tienden a concentrarse en baja antigüedad (<20 meses) y cargos totales moderados a altos, confirmando patrones de riesgo.



Matrices de Confusión:





Regresión Logística:





Random Forest:





Insight: Regresión Logística detecta mejor las cancelaciones (recall 54.28%) que Random Forest (48.40%), aunque ambos modelos fallan en identificar muchos casos de Evasión=1 debido al desbalance.



Variables Relevantes:





Regresión Logística:





Random Forest:





Insight: Variables clave incluyen Antigüedad (menor antigüedad aumenta cancelación), Cargos_Totales (costos altos incrementan riesgo), y Contrato_Month-to-month (mayor riesgo frente a contratos a largo plazo).

Instrucciones para Ejecutar el Cuaderno

Para reproducir el análisis en data_preparation.ipynb, sigue estos pasos:

Requisitos





Entorno: Google Colab (recomendado) o un entorno local con Python 3.7+.



Librerías necesarias: Instala las siguientes librerías usando pip:

pip install pandas numpy seaborn matplotlib plotly scikit-learn

En Colab, estas librerías suelen estar preinstaladas, pero verifica ejecutando:

import pandas, numpy, seaborn, matplotlib, plotly, sklearn

Pasos para Ejecutar





Clonar el repositorio (si usas un entorno local):

git clone https://github.com/carlo55anchez/TelecomX-Analysis2.git
cd TelecomX-Analysis2



Abrir el cuaderno:





En Colab: Sube notebooks/data_preparation.ipynb a Google Colab o ábrelo directamente desde el enlace en el repositorio.



En local: Usa Jupyter Notebook o VS Code:

jupyter notebook notebooks/data_preparation.ipynb



Cargar los datos:





El cuaderno carga automáticamente TelecomX_Data_standardized.csv desde:

url = 'https://raw.githubusercontent.com/carlo55anchez/TelecomX-Analysis2/main/data/TelecomX_Data_standardized.csv'
df = pd.read_csv(url)



Si trabajas localmente, asegúrate de que el archivo esté en data/ o usa la URL proporcionada.



Ejecutar las celdas:





Ejecuta las celdas en orden (Shift+Enter en Colab/Jupyter).



Las visualizaciones se generarán automáticamente, y los gráficos se guardarán en la carpeta img/ (Etapa 12).



Descargar gráficos (opcional):





En Colab, los gráficos se guardan en img/. Descárgalos desde el panel de archivos o usa:

import shutil
from google.colab import files
shutil.make_archive('img_archive', 'zip', 'img')
files.download('img_archive.zip')

Notas





Asegúrate de tener una conexión a internet para cargar el dataset desde la URL.



Si encuentras errores, verifica las versiones de las librerías o revisa los mensajes de error en Colab.

Conclusión

El proyecto TelecomX-Analysis2 proporciona un análisis completo para predecir la cancelación de clientes en TelecomX, identificando factores clave como Antigüedad, Cargos_Totales, y Contrato_Month-to-month. Las estrategias de retención propuestas (contratos a largo plazo, mejora de fibra óptica, gestión de costos) se derivan de los resultados. El cuaderno data_preparation.ipynb y las visualizaciones en img/ ofrecen una base sólida para explorar y extender el análisis, con recomendaciones para mejoras como balanceo de clases (SMOTE) o optimización de hiperparámetros.
