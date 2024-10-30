# -*- coding: utf-8 -*-
"""
UNIVERSIDAD DE LAS AMÉRICAS
FACULTAD DE INGENIERÍA Y CIENCIAS APLICADAS
INGENIERÍA DE SOFTWARE
INTELIGENCIA ARTIFICIAL 1

REGRESIÓN POLINÓMICA
POR: JOE CORDERO
"""

## Importación de las librerías necesarias para el preprocesamiento de datos y visualización
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carga del conjunto de datos desde un archivo CSV
# El dataset contiene información de las ventas de helados en función de la temperatura
# Fuente del dataset: https://www.kaggle.com/datasets/mirajdeepbhandari/polynomial-regression
dataset = pd.read_csv("./IA 1/Ice_cream_selling_data_ordered.csv")

# Definición de la variable independiente (temperatura en grados centígrados)
X = dataset.iloc[:, 0:1].values

# Definición de la variable dependiente (ventas de helados)
y = dataset.iloc[:, 1:2].values

## Aplicación de la regresión polinómica con grado 2
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)  # El grado del polinomio es 2
X_poly = poly_reg.fit_transform(X)  # Genera las variables b0, X1 y X1^2

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

### Visualización del modelo de regresión polinómica ajustado
plt.scatter(X, y, color="red")  # Puntos reales del dataset
plt.plot(X, lin_reg2.predict(X_poly), color="blue")  # Línea del modelo de regresión polinómica
### Comentario adicional: En caso de otro conjunto de datos, usar
### plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color="blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Temperatura en grados centígrados")
plt.ylabel("Ventas de helados")
plt.show()

## Mejora del modelo incrementando el grado del polinomio
## Ajuste para que el polinomio sea de grado 3
## Aplicación de la regresión polinómica con grado 3
poly_reg = PolynomialFeatures(degree=3)  # Incremento del grado a 3
X_poly = poly_reg.fit_transform(X)  # Generación de las variables b0, X1, X1^2, y X1^3

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

### Visualización del modelo de regresión polinómica ajustado con grado 3
plt.scatter(X, y, color="red")  # Puntos reales del dataset
plt.plot(X, lin_reg2.predict(X_poly), color="blue")  # Línea del modelo ajustado con grado 3
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Temperatura en grados centígrados")
plt.ylabel("Ventas de helados")
plt.show()

## Mejora del modelo incrementando el grado del polinomio
## Ajuste para que el polinomio sea de grado 4
## Aplicación de la regresión polinómica con grado 4
poly_reg = PolynomialFeatures(degree=4)  # Incremento del grado a 4
X_poly = poly_reg.fit_transform(X)  # Generación de las variables b0, X1, X1^2, y X1^3

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

### Visualización del modelo de regresión polinómica ajustado con grado 4
plt.scatter(X, y, color="red")  # Puntos reales del dataset
plt.plot(X, lin_reg2.predict(X_poly), color="blue")  # Línea del modelo ajustado con grado 4
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Temperatura en grados centígrados")
plt.ylabel("Ventas de helados")
plt.show()
