# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:12:17 2024

@author: joema
REGRESION POLINÓMICA

1.- Debo cachar en excel si es polinomica, o sea una curva 
Aqui es mejor trabajar con matrices
Al poner .2f puedo saber si las variables tienen decimales
Aqui no se divide en train y test ya que no hay registros repetidos
Tampoco debemos escalar (hacer que sean comparables-normalizar o estandarizar)
ya que no hay nan, no hay repetidos rampoco
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("IA 1/Position_Salaries.csv")

##Aqui la X son los niveles
X = dataset.iloc[:, 1:2].values

##Aqui la y son los salarios
y = dataset.iloc[:, 2:3].values

##Aqui aplico el algoritmo de regresion lineal, la cual solo es para ver como funciona
##En realidad solo debo hacer la polinómica
from sklearn.linear_model import LinearRegression
##Aqui se ajustan los datos con el fit
lin_reg = LinearRegression()
lin_reg.fit(X, y)

##Aqui aplico la regresion polinómica
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) ##el grado por defecto siempre es 2
X_poly = poly_reg.fit_transform(X) ##Aqui ya me dió las variables b0, X1 y X1^2

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

### Aqui grafico la regresion lineal
##SOLO ES PARA VER, NO HAY QUE HACERLA
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg.predict(X), color="blue")
plt.title("Modelo de Regresión Linear")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo en ($)")
plt.show()

### Despliegue de la información con visualización del modelo de regresión polinómica
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg2.predict(X_poly), color="blue")
### Si tuviera otro conjunto de datos
### plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color="blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo en ($)")
plt.show()

##El resultado del grafico es correcto, pero los espacios entre cada punto es muy amplio, por lo cual
##vamos a crear mas puntos, cambiar el rango y ajustar.
##Aqui creamos la grilla con los puntos adicionales
X_grid = np.arange(min(X[:, 0]), max(X[:, 0]), 0.1, dtype=float)
X_grid = X_grid.reshape((len(X_grid), 1))

##Graficamos
plt.scatter(X, y, color="red")
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color="blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo en ($)")
plt.show()
## el modelo reusltante todavia falta afinar, por lo cual tenemos que afinarlo mas
##Antes de eso solo vamos a predecir e imprimir cuanto deberia ganar con el modelo actual
##en el nivel 6.5
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))

##Vamos a afinar para que ahora el polinomio sea de grado 3
##Aqui aplico la regresion polinómica
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3) ##el grado por defecto siempre es 2
X_poly = poly_reg.fit_transform(X) ##Aqui ya me dió las variables b0, X1 y X1^2

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

### Despliegue de la información con visualización del modelo de regresión polinómica
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg2.predict(X_poly), color="blue")
### Si tuviera otro conjunto de datos
### plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color="blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo en ($)")
plt.show()
##El modelo ha mejorado pero aun se puede afinar mucho mas
##calculamos la grilla y predecimos
X_grid = np.arange(min(X[:, 0]), max(X[:, 0]), 0.1, dtype=float)
X_grid = X_grid.reshape((len(X_grid), 1))
lin_reg2.predict(poly_reg.fit_transform([[6.5]])) ##Aqui ya se evidencia como se ha ajustado el sueldo

##Cambiamos a grado 4
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) ##el grado por defecto siempre es 2
X_poly = poly_reg.fit_transform(X) ##Aqui ya me dió las variables b0, X1 y X1^2

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

### Despliegue de la información con visualización del modelo de regresión polinómica
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg2.predict(X_poly), color="blue")
### Si tuviera otro conjunto de datos
### plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color="blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo en ($)")
plt.show()
##El modelo ha mejorado pero aun se puede afinar mucho mas
##calculamos la grilla y predecimos
X_grid = np.arange(min(X[:, 0]), max(X[:, 0]), 0.1, dtype=float)
X_grid = X_grid.reshape((len(X_grid), 1))
lin_reg2.predict(poly_reg.fit_transform([[6.5]])) ##Aqui ya se evidencia como se ha ajustado el sueldo


