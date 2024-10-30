# -*- coding: utf-8 -*-
"""
UNIVERSIDAD DE LAS AMÉRICAS
FACULTAD DE INGENIERÍA Y CIENCIAS APLICADAS
INGENIERÍA DE SOFTWARE
INTELIGENCIA ARTIFICIAL 1

REGRESIÓN POLINÓMICA
POR: JOE CORDERO
"""

## Importación de librerías necesarias para el preprocesamiento de datos y visualización
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carga del conjunto de datos desde un archivo CSV
# Dataset con información de las ventas de helados en función de la temperatura
# Fuente: https://www.kaggle.com/datasets/mirajdeepbhandari/polynomial-regression
dataset = pd.read_csv("./IA 1/Ice_cream_selling_data_ordered.csv")

# Definición de las variables independientes (temperatura en grados centígrados)
X = dataset.iloc[:, 0:1].values

# Definición de la variable dependiente (Ventas de helados)
y = dataset.iloc[:, 1:2].values

##Aqui aplico la regresion polinómica
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) ##el grado por defecto siempre es 2
X_poly = poly_reg.fit_transform(X) ##Aqui ya me dió las variables b0, X1 y X1^2

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

### Despliegue de la información con visualización del modelo de regresión polinómica
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg2.predict(X_poly), color="blue")
### Si tuviera otro conjunto de datos
### plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color="blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Temperatura en grados centígrados")
plt.ylabel("Ventas de helados")
plt.show()


##Vamos a afinar para que ahora el polinomio sea de grado 3
##Aqui aplico la regresion polinómica
poly_reg = PolynomialFeatures(degree = 3) ##el grado por defecto siempre es 2
X_poly = poly_reg.fit_transform(X) ##Aqui ya me dió las variables b0, X1 y X1^2

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

### Despliegue de la información con visualización del modelo de regresión polinómica
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg2.predict(X_poly), color="blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Temperatura en grados centígrados")
plt.ylabel("Ventas de helados")
plt.show()

##Vamos a afinar para que ahora el polinomio sea de grado 4
##Aqui aplico la regresion polinómica
poly_reg = PolynomialFeatures(degree = 4) ##el grado por defecto siempre es 2
X_poly = poly_reg.fit_transform(X) ##Aqui ya me dió las variables b0, X1 y X1^2

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

### Despliegue de la información con visualización del modelo de regresión polinómica
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg2.predict(X_poly), color="blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Temperatura en grados centígrados")
plt.ylabel("Ventas de helados")
plt.show()

##Vamos a afinar para que ahora el polinomio sea de grado 5
##Aqui aplico la regresion polinómica
poly_reg = PolynomialFeatures(degree = 5) ##el grado por defecto siempre es 2
X_poly = poly_reg.fit_transform(X) ##Aqui ya me dió las variables b0, X1 y X1^2

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

### Despliegue de la información con visualización del modelo de regresión polinómica
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg2.predict(X_poly), color="blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Temperatura en grados centígrados")
plt.ylabel("Ventas de helados")
plt.show()

##Vamos a afinar para que ahora el polinomio sea de grado 6
##Aqui aplico la regresion polinómica
poly_reg = PolynomialFeatures(degree = 6) ##el grado por defecto siempre es 2
X_poly = poly_reg.fit_transform(X) ##Aqui ya me dió las variables b0, X1 y X1^2

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

### Despliegue de la información con visualización del modelo de regresión polinómica
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg2.predict(X_poly), color="blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Temperatura en grados centígrados")
plt.ylabel("Ventas de helados")
plt.show()


##Metodo para ver hasta que grado se puede seguir afinando
##el cambio y variabilidad de los resultados nos muestra hasta que punto
##se puede seguir mejorando, mientras menos cambios entonces quiere decir que
## no tiene caso seguir ajustando el grado del polinomio

from sklearn.metrics import mean_squared_error, r2_score

# Función para ajustar el modelo y calcular R² y MSE
def ajustar_modelo_polynomial(degree):
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    
    # Predicciones del modelo
    y_pred = lin_reg.predict(X_poly)
    
    # Cálculo de R² y MSE
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    # Mostrar gráfica y métricas
    plt.scatter(X, y, color="red")
    plt.plot(X, y_pred, color="blue")
    plt.title(f"Modelo de Regresión Polinómica (Grado {degree})")
    plt.xlabel("Temperatura en grados centígrados")
    plt.ylabel("Ventas de helados")
    plt.show()
    
    print(f"Grado {degree} - R²: {r2:.4f}, MSE: {mse:.4f}")

# Probar modelos de grados 2 a 6
for grado in range(2, 10):
    ajustar_modelo_polynomial(grado)
