# -*- coding: utf-8 -*-
"""
UNIVERSIDAD DE LAS AMÉRICAS
FACULTAD DE INGENIERÍA Y CIENCIAS APLICADAS
INGENIERÍA DE SOFTWARE
INTELIGENCIA ARTIFICIAL 1

REGRESIÓN LINEAL SIMPLE
POR: JOE CORDERO
"""

## Importación de librerías necesarias para el preprocesamiento de datos y visualización
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carga del conjunto de datos desde un archivo CSV
# Dataset con información del la capacidad pulmunar de una persona en relación a las horas de ejercicio semanal realizadas.
# Fuente: Herramienta de inteligencia artificial
dataset = pd.read_csv("./IA 1/Data.csv")

# Definición de la variable independiente (Horas de ejercicio semanal)
X = dataset.iloc[:, 0].values

# Definición de la variable dependiente (Capacidad Pulmonar en litros)
y = dataset.iloc[:, 1].values

## Imputación de valores faltantes (NaN) en las variables independientes y dependientes
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = "median")

# Tanto 'X' como 'y' contienen una sola columna, pero 'SimpleImputer' requiere un array bidimensional,
# por lo que se realiza la conversión correspondiente
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Ajuste e imputación de valores faltantes en la variable independiente 'X'
imputer_X = imputer.fit(X)
X = imputer_X.transform(X)

# Ajuste e imputación de valores faltantes en la variable dependiente 'y'
imputer_y = imputer.fit(y)
y = imputer_y.transform(y)

## División del conjunto de datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Se divide el conjunto de datos con un 80% para entrenamiento y un 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

## Creación y entrenamiento del modelo de regresión lineal
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)

## Predicción de los valores de 'y' para el conjunto de prueba
y_pred = regression.predict(X_test)

## Visualización de los resultados de la predicción
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_test, regression.predict(X_test), color = "blue")
plt.title("Capacidad Pulmonar vs. Horas semanales de ejercicio")
plt.xlabel("Horas semanales de ejercicio")
plt.ylabel("Capacidad Pulmonar")
plt.show()