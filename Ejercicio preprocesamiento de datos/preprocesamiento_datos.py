# -*- coding: utf-8 -*-
"""
Editor de Spyder

PREPROCESAMIENTO DE DATOS
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("C:/Users/laboratorio/Downloads/Data.csv")

## saca todas las filas y las columnas menos una
X = dataset.iloc[:, :-1].values

## Saca todas las filas y la columna numero 3
y = dataset.iloc[:, 3].values

## Libreria para limpieza e imputación de datos
from sklearn.impute import SimpleImputer

## imputar es para estandarizar y tambien para normalizar
## aqui los NAN se llenan de la media
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

## cuando el set de datos es pequeño entonces sirve la media
## cuando el set es muy grande se usa la mediana

## fit dice que toma las columnas donde se puede aplicar la estrategia de la media (en la 0 no se puede porque es texto)
## EN PYTHON es n-1 al momento de poner el indice de la columna
imputer = imputer.fit(X[:, 1:3])
## aqui le dice que en los valores de X transforme los NAN a la media
X[:, 1:3] = imputer.transform(X[:, 1:3])

## PREPROCESAR PARA HACER DE STRING A NUMERO
from sklearn import preprocessing

## aqui transformamos de variable categorica a numerica
## si las variables categoricas tiene mas de dos opciones entonces hay que hacerle labelencoder y tambien one hot encoder
labelencoder_X = preprocessing.LabelEncoder()

X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

## Aqui pasamos a variables dummy
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])], remainder='passthrough')


## variable Dummy para variable X
X = np.array(onehotencoder.fit_transform(X), dtype=float) ## Genera una matriz de ceros y unos

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

## Ya acabamos el transform de ETL

## Ahora vamos con el entrenamiento y testing
from sklearn.model_selection import train_test_split

# Dividir el dataset en conjunto de entrenamiento y testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## Escalado de variables
## Procesamos la estandarización
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)  ## el conjunto X

## aqui ya está estandarizada, y eso sirve para que se hagan comparables












