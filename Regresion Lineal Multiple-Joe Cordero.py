# -*- coding: utf-8 -*-
"""
UNIVERSIDAD DE LAS AMÉRICAS
FACULTAD DE INGENIERÍA Y CIENCIAS APLICADAS
INGENIERÍA DE SOFTWARE
INTELIGENCIA ARTIFICIAL 1

REGRESIÓN LINEAL MÚLTIPLE
POR: JOE CORDERO
"""

## Importación de librerías necesarias para el preprocesamiento de datos y visualización
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carga del conjunto de datos desde un archivo CSV
# Dataset con información del índice de performance segun factores que afectan al estudiante
# Fuente: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression?resource=download
dataset = pd.read_csv("./IA 1/Student_Performance.csv")

# Definición de las variables independientes (factores que afectan el índice de performance)
X = dataset.iloc[:, :5].values  # Se toman las primeras cuatro columnas como predictores

# Definición de la variable dependiente (Índice de Performance)
y = dataset.iloc[:, 5].values  # Se selecciona la quinta columna como la variable dependiente

## Codificación de la variable categórica "Actividades Extracurriculares"
## Aquí se transforma la columna X[:,2], que es una variable categórica (Yes/No),
## en valores numéricos usando LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

labelencoder_X = preprocessing.LabelEncoder()

# Convertir "Yes" y "No" en 1 y 0 respectivamente
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

## División del conjunto de datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Se divide el conjunto de datos en 80% para entrenamiento y 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## Creación y entrenamiento del modelo de regresión lineal múltiple
from sklearn.linear_model import LinearRegression

# Instanciar el modelo de regresión lineal
regression = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
regression.fit(X_train, y_train)

## Predicción de los valores de 'y' para el conjunto de prueba
# Predecir el índice de performance para los datos de prueba
y_pred = regression.predict(X_test)

## Selección de variables mediante el método de eliminación hacia atrás
import statsmodels.api as sm 

# Añadir una columna de unos a X (para el término constante o intercepto)
X = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)

# Definir el nivel de significación para la eliminación hacia atrás
SL = 0.05

# Seleccionar todas las columnas inicialmente (todas las variables predictoras)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]  # Incluyendo la columna de unos

# Ajustar el modelo de regresión OLS (Mínimos Cuadrados Ordinarios) con todas las variables
regression_OLS = sm.OLS(y, X_opt).fit()

# Mostrar el resumen del modelo para evaluar los p-valores de las variables
regression_OLS.summary()

## En este caso en particular, se evidenció que todas las variables son significantes para el indice de performance
## por lo cual se han mantenido todas las variables inependientes

## Visualización de los resultados de la predicción
# Gráfico de dispersión: valores reales vs. predicciones con línea de identidad
plt.scatter(y_test, y_pred, color="blue")  # Puntos de las predicciones vs. valores reales
# Línea de identidad (y_test = y_pred)
# Linea que nos permite evaluar cuán cerca están las predicciones del modelo de los valores reales.
# Si los puntos en el gráfico están cerca de la línea de identidad, el modelo está haciendo predicciones precisas.
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red")
plt.title("Predicciones vs. Valores Reales del Indice de Performance")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.show()


##esto sirve para cuando no se reconocen los valores como numeros
X = X.astype(int)  # o X = X.astype(float) si necesitas decimales

'''
##Este es el analisis para la regresion lineal multiple

cinco elementos clave en la aplicación de la Regresión Lineal Múltiple (RLM):

1. Definición de Variables Independientes y Dependientes
Variable Dependiente (Y): Es la variable que se quiere predecir o explicar a partir de otras variables. En RLM, esta variable debe ser continua (por ejemplo, ingresos, ventas, peso).
Variables Independientes (X1, X2, X3, ...): Son las variables predictoras que se cree afectan la variable dependiente. Por ejemplo, en un modelo que predice el peso de una persona, las variables independientes podrían ser su edad, altura y nivel de actividad física.
Importancia: La elección de estas variables es fundamental, ya que deben estar relacionadas con el fenómeno que se intenta predecir. Un modelo sin las variables adecuadas tendrá baja precisión.

2. Suposiciones de la Regresión Lineal Múltiple
Para que el modelo sea válido, debe cumplir con varias suposiciones clave:
Linealidad: La relación entre las variables independientes y la dependiente debe ser lineal. Esto significa que el cambio en la variable dependiente debe ser proporcional a los cambios en las independientes.
No Colinealidad (Multicolinealidad): Se espera que las variables independientes no estén altamente correlacionadas entre sí. Si existe una alta correlación entre dos o más variables independientes, puede distorsionar los coeficientes de regresión y hacer que el modelo sea menos interpretable.
Detección: Para detectar la multicolinealidad, se suele usar el factor de inflación de la varianza (VIF). Un VIF alto indica colinealidad.
Homocedasticidad: Los errores o residuos (diferencias entre los valores observados y los predichos) deben tener una varianza constante a lo largo de todos los niveles de las variables independientes.
Problemas de Heterocedasticidad: Si la varianza de los errores cambia, puede dar lugar a inferencias engañosas. La prueba de Breusch-Pagan es común para detectar heterocedasticidad.
Normalidad de los Errores: Los errores del modelo deberían seguir una distribución normal, lo que se verifica mediante gráficos Q-Q o histogramas de residuos. Esto asegura que las predicciones del modelo sean válidas y fiables para inferencia estadística.
Independencia de los Errores: Los errores deben ser independientes unos de otros. Esto es particularmente importante en datos secuenciales o dependientes en el tiempo (series temporales).


3. Preprocesamiento de los Datos
El preprocesamiento es crucial para asegurar que los datos estén en el formato adecuado para el modelo y que los resultados sean precisos y válidos.
Codificación de Variables Categóricas: Si tienes variables categóricas (por ejemplo, género, tipo de actividad), necesitas convertirlas en una representación numérica. Esto se logra mediante técnicas como OneHotEncoder o LabelEncoder.
Escalado y Normalización de Variables: Aunque no siempre es obligatorio, normalizar o escalar los datos puede ayudar en algunos modelos. Por ejemplo, si tienes variables en diferentes unidades, normalizarlas asegura que ninguna variable tenga una influencia desproporcionada.
Tratamiento de Valores Atípicos y Faltantes:
Valores Atípicos (outliers): Pueden distorsionar los resultados. Se detectan mediante gráficos de dispersión o de caja y pueden manejarse eliminándolos o transformándolos.
Valores Faltantes: Se pueden imputar (rellenar) usando métodos como la media, mediana o imputación avanzada con SimpleImputer de scikit-learn.


4. Construcción y Ajuste del Modelo
Una vez que se han procesado los datos, el siguiente paso es construir y ajustar el modelo RLM.
Entrenamiento del Modelo: Esto implica ajustar el modelo a los datos, es decir, encontrar los coeficientes de regresión que minimicen el error en las predicciones.
Selección de Variables: Durante el ajuste del modelo, es útil reducir el número de variables predictoras a las más significativas para mejorar la precisión y simplicidad del modelo.
Eliminación hacia Atrás: Consiste en construir el modelo con todas las variables y eliminar las menos significativas iterativamente, basado en el p-valor.
Métodos de Regularización: Ridge y Lasso son métodos que agregan una penalización a los coeficientes del modelo para reducir la complejidad y manejar la multicolinealidad.


5. Evaluación del Modelo
Evaluar el modelo asegura que funcione adecuadamente y pueda generalizarse a nuevos datos. Los principales métodos incluyen:
Coeficiente de Determinación (R²): Mide qué tan bien el modelo explica la variabilidad de la variable dependiente. Un valor más cercano a 1 indica un mejor ajuste.
Error Cuadrático Medio (MSE): Promedio de los errores al cuadrado. Un MSE bajo indica que las predicciones están cerca de los valores reales.
Pruebas de Significancia para los Coeficientes: Evaluar los p-valores de cada variable independiente permite ver cuáles son estadísticamente significativas.
Análisis de los Residuos: Para confirmar si el modelo cumple con las suposiciones, se revisan los residuos. Un gráfico Q-Q de los residuos puede verificar la normalidad, y un gráfico de dispersión de los residuos contra los valores predichos puede confirmar la homocedasticidad.
'''