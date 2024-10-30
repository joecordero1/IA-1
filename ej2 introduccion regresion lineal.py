# -*- coding: utf-8 -*-
"""
Editor de Spyder

Regresion linea simple DE DATOS
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("./IA 1/Salary_Data.csv")

## saca todas las filas y las columnas menos una
## [Colmuna inicial : Colmuna Final, Filas]

X = dataset.iloc[:, :1].values

## Saca todas las filas y la columna numero 3
y = dataset.iloc[:, 1].values

## Ya acabamos el transform de ETL

## Ahora vamos con el entrenamiento y testing
from sklearn.model_selection import train_test_split

# Dividir el dataset en conjunto de entrenamiento y testing
##aQUI TOMAMOS UN TERCIO de los registros
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

##  REGRESION LINEAL
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)

## Aqui predigo, se lo hace con test xq aqui se verifica que en verdad el modelo haya entrenado y aprendido
y_pred = regression.predict(X_test)

##este es el grafico de la prediccion de train
##hace los puntos 
plt.scatter(X_train, y_train, color = "red")
##hace la recta
plt.plot(X_train, regression.predict(X_train), color = "blue")
##titulo
plt.title("Sueldo vs. Años de Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo ($)")
plt.show()


##este es el grafico de la prediccion de test
##hace los puntos 
plt.scatter(X_test, y_test, color = "red")
##hace la recta
plt.plot(X_test, regression.predict(X_test), color = "blue")
##titulo
plt.title("Sueldo vs. Años de Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo ($)")
plt.show()

## la media es susceptible a los outliers
## cuando la curva es normal, la media y la mediana son iguales
## imputar es pponer valores donde no hay, con estrategias de vecino cercano, media y mediana
## la estrategia para escalar son estandarizar y normalizar
## estandarizar son x max y x min
## normalizar es varianza
## la estandarizacion nor sirve para que las variables puedan ser comparables, es decir, entre 0 y 1
## se debe escalar el e nrenamiento y prueba para quye puedan ser comparables

## la recta es el model original y los puntos son las predicciones
##La recta creciente es directamente proporcional, y la decrecente es inversamente proporcional
##predecir es estadistica inferencial
