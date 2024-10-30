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