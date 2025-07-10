# TelecomX-Analysis2: Análisis Predictivo de Evasión de Clientes

## Propósito del Análisis
El proyecto `TelecomX-Analysis2` tiene como objetivo principal predecir la cancelación de clientes (churn, representado por la variable `Evasión`) en la empresa TelecomX, utilizando un conjunto de datos con 7267 muestras y 31 variables (tras preprocesamiento). El análisis identifica los factores clave que influyen en la cancelación y propone estrategias de retención basadas en modelos predictivos (Regresión Logística y Random Forest). Este trabajo combina análisis exploratorio de datos (EDA), preparación de datos, modelado, evaluación, e interpretación de resultados para ofrecer insights accionables que mejoren la retención de clientes.

## Estructura del Proyecto
El repositorio está organizado en las siguientes carpetas y archivos:

- **data/**: Contiene el archivo de datos preprocesado.
  - `TelecomX_Data_standardized.csv`: Conjunto de datos con 7267 muestras, incluyendo variables numéricas y categóricas codificadas.
- **notebooks/**: Contiene el cuaderno principal del análisis.
  - `data_preparation.ipynb`: Notebook con todo el proceso, desde la carga de datos hasta el informe final.
- **img/**: Contiene las visualizaciones generadas durante el análisis.
  - `correlation_matrix.png`: Matriz de correlación de variables.
  - `boxplots_evasion.png`: Boxplots de `Antigüedad` y `Cargos_Totales` por `Evasión`.
  - `scatter_evasion.png`: Scatter plot de `Antigüedad` vs. `Cargos_Totales` coloreado por `Evasión`.
  - `confusion_matrix_lr.png`: Matriz de confusión de Regresión Logística.
  - `confusion_matrix_rf.png`: Matriz de confusión de Random Forest.
  - `feature_importance_lr.png`: Top 10 variables relevantes (Regresión Logística).
  - `feature_importance_rf.png`: Top 10 variables relevantes (Random Forest).

## Proceso de Preparación de Datos
El proceso de preparación de datos se realizó en varias etapas para asegurar que el conjunto estuviera listo para el modelado predictivo:

### Clasificación de Variables
- **Numéricas**: `Antigüedad`, `Cargos_Mensuales`, `Cargos_Totales`, `Cargos_Diarios`.
- **Categóricas**: Incluyen `Contrato` (Month-to-month, One year, Two year), `Servicio_Internet` (DSL, Fiber optic, No), `Método_Pago`, `Facturación_Sin_Papel`, entre otras.
- **Variable objetivo**: `Evasión` (0 = Activo, 1 = Canceló), con un desbalance del 74.28% (activos) y 25.72% (cancelados).

### Etapas de Normalización y Codificación
1. **Eliminación de columnas irrelevantes**: Se eliminó `ID_Cliente` por no aportar valor predictivo.
2. **Codificación de variables categóricas**: Se aplicó codificación one-hot a variables categóricas (por ejemplo, `Contrato_Month-to-month`, `Contrato_Two year`), generando un DataFrame con 31 columnas (`df_encoded`).
3. **Estandarización**: Las variables numéricas (`Antigüedad`, `Cargos_Mensuales`, `Cargos_Totales`, `Cargos_Diarios`) se estandarizaron con `StandardScaler` (media=0, desviación estándar=1) para Regresión Logística, asegurando que las magnitudes dispares (por ejemplo, `Cargos_Totales`: 18.8-8684.8) no sesgaran los coeficientes. Random Forest no requirió estandarización.

### Separación de Datos
- **División**: El conjunto se dividió en 80% entrenamiento (5813 muestras) y 20% prueba (1454 muestras) usando `train_test_split` con `stratify=y` para mantener la proporción de clases (74.28% `Evasión=0`, 25.72% `Evasión=1`).
- **Justificación**: La división 80/20 es adecuada para un conjunto de 7267 muestras, proporcionando suficientes datos para entrenamiento y prueba. La estratificación asegura que el desbalance se mantenga en ambos conjuntos.

## Justificaciones de las Decisiones de Modelado
Se entrenaron dos modelos para predecir `Evasión`:

1. **Regresión Logística**:
   - **Razón**: Modelo simple, interpretable, adecuado para relaciones lineales. Requiere estandarización debido a la sensibilidad a la escala de las variables.
   - **Rendimiento**: Accuracy 80.26%, recall para `Evasión=1` 54.28%, F1-score 0.59. Sin evidencia de overfitting (entrenamiento: 80.70%, prueba: 80.26%), pero posible underfitting por simplicidad.
   - **Limitación**: Puede no capturar relaciones no lineales, afectando el recall.

2. **Random Forest**:
   - **Razón**: Modelo robusto, no requiere estandarización, adecuado para relaciones no lineales y desbalance moderado.
   - **Rendimiento**: Accuracy 78.13%, recall para `Evasión=1` 48.40%, F1-score 0.53. Fuerte overfitting (entrenamiento: 99.11%, prueba: 78.13%), debido a la alta complejidad (árboles profundos sin restricciones).
   - **Limitación**: El desbalance y la falta de ajuste de hiperparámetros redujeron su desempeño.

**Desbalance**: El 25.72% de `Evasión=1` limitó el recall en ambos modelos. Técnicas como SMOTE o `class_weight='balanced'` se sugirieron para mejorarlo. Regresión Logística fue el mejor modelo, pero ambos necesitan optimización (por ejemplo, ajuste de hiperparámetros con `GridSearchCV`).

## Ejemplos de Gráficos e Insights del EDA
El análisis exploratorio de datos (EDA) proporcionó los siguientes insights, respaldados por visualizaciones:

1. **Matriz de Correlación**:
   - **Gráfico**: ![Matriz de Correlación](https://raw.githubusercontent.com/carlo55anchez/TelecomX-Analysis2/main/img/correlation_matrix.png)
   - **Insight**: `Antigüedad` tiene una correlación negativa con `Evasión` (-0.35), indicando que clientes con mayor tiempo tienden a permanecer. `Cargos_Totales` y `Contrato_Month-to-month` tienen correlaciones positivas (0.20 y 0.41), sugiriendo mayor riesgo de cancelación.

2. **Boxplots de Antigüedad y Cargos Totales**:
   - **Gráfico**: ![Boxplots](https://raw.githubusercontent.com/carlo55anchez/TelecomX-Analysis2/main/img/boxplots_evasion.png)
   - **Insight**: Clientes que cancelan tienen menor `Antigüedad` (mediana ~10 meses vs. ~38 meses para activos) y mayores `Cargos_Totales` (mediana ~1390 vs. ~1160 para activos).

3. **Scatter Plot: Antigüedad vs. Cargos Totales**:
   - **Gráfico**: ![Scatter Plot](https://raw.githubusercontent.com/carlo55anchez/TelecomX-Analysis2/main/img/scatter_evasion.png)
   - **Insight**: Clientes con baja antigüedad (<20 meses) y altos cargos totales (>2000) son más propensos a cancelar.

4. **Matrices de Confusión**:
   - **Gráficos**: 
     - ![Matriz de Confusión - Regresión Logística](https://raw.githubusercontent.com/carlo55anchez/TelecomX-Analysis2/main/img/confusion_matrix_lr.png)
     - ![Matriz de Confusión - Random Forest](https://raw.githubusercontent.com/carlo55anchez/TelecomX-Analysis2/main/img/confusion_matrix_rf.png)
   - **Insight**: Regresión Logística detecta mejor los casos de `Evasión=1` (recall 54.28%) que Random Forest (48.40%), aunque ambos tienen falsos negativos debido al desbalance.

5. **Variables Relevantes**:
   - **Gráficos**:
     - ![Importancia - Regresión Logística](https://raw.githubusercontent.com/carlo55anchez/TelecomX-Analysis2/main/img/feature_importance_lr.png)
     - ![Importancia - Random Forest](https://raw.githubusercontent.com/carlo55anchez/TelecomX-Analysis2/main/img/feature_importance_rf.png)
   - **Insight**: `Antigüedad`, `Cargos_Totales`, `Contrato_Month-to-month`, `Servicio_Internet_Fiber optic`, y `Facturación_Sin_Papel` son los factores más influyentes en la cancelación.

## Instrucciones para Ejecutar el Cuaderno
Para reproducir el análisis en Google Colab, sigue estos pasos:

1. **Instalar bibliotecas**:
   - Ejecuta la siguiente celda en Colab para instalar las dependencias necesarias:
     ```python
     !pip install pandas numpy seaborn matplotlib plotly scikit-learn
     ```
   - Bibliotecas requeridas: `pandas`, `numpy`, `seaborn`, `matplotlib`, `plotly.express`, `scikit-learn`.

2. **Cargar los datos**:
   - El notebook carga los datos directamente desde la URL cruda:
     ```python
     url = 'https://raw.githubusercontent.com/carlo55anchez/TelecomX-Analysis2/main/data/TelecomX_Data_standardized.csv'
     df = pd.read_csv(url)
     ```
   - Asegúrate de que el archivo `TelecomX_Data_standardized.csv` esté disponible en `data/`.

3. **Ejecutar el notebook**:
   - Abre `notebooks/data_preparation.ipynb` en Google Colab.
   - Ejecuta todas las celdas en orden (desde la carga de datos hasta la generación del informe y gráficos).
   - Los gráficos se guardan en la carpeta `img/` del entorno de Colab y pueden descargarse para el repositorio.

4. **Notas**:
   - El notebook está diseñado para ejecutarse en Google Colab, pero puede adaptarse a entornos locales instalando las bibliotecas mencionadas.
   - Verifica que la conexión a internet permita acceder a la URL de GitHub.

## Conclusión
El proyecto `TelecomX-Analysis2` proporciona un análisis completo de la cancelación de clientes en TelecomX, identificando factores clave como baja `Antigüedad`, altos `Cargos_Totales`, y contratos `Month-to-month`. Las estrategias de retención propuestas (contratos a largo plazo, mejora de fibra óptica, gestión de costos) son accionables y están respaldadas por los resultados. El notebook `data_preparation.ipynb` y las visualizaciones en `img/` ofrecen una base sólida para futuros análisis o implementaciones.
