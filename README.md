# IA-1
## Como diferenciar entre los tres tipos de regresiones.
Para determinar qué tipo de regresión es la más adecuada según el conjunto de datos, aquí tienes algunas pautas que te ayudarán a elegir entre regresión lineal simple, múltiple y polinómica:

1. Regresión Lineal Simple
Descripción: Se usa cuando tienes una relación lineal entre dos variables: una variable independiente (predictora) y una variable dependiente (respuesta).
Cómo identificarla:
Solo tienes una variable independiente y una variable dependiente en el conjunto de datos.
Los datos parecen tener una relación lineal (directa o inversa) al graficarlos en un plano.
Ejemplo en tu código: La relación entre horas de ejercicio semanal y capacidad pulmonar es un caso típico para probar regresión lineal simple si observas una relación directa en el gráfico.
2. Regresión Lineal Múltiple
Descripción: Aplica cuando tienes múltiples variables independientes que pueden afectar la variable dependiente y deseas explorar sus efectos combinados.
Cómo identificarla:
Tienes dos o más variables independientes (predictoras) y una variable dependiente.
Ideal cuando quieres ver cómo varios factores contribuyen a una variable de respuesta. Por ejemplo, salario puede depender de años de experiencia, nivel de educación y horas de capacitación.
Es común probarla cuando las relaciones entre variables independientes y dependientes parecen lineales (no curvas).
Evaluación de multicolinealidad: En regresión múltiple, verifica que no haya alta correlación entre variables independientes. Si están muy correlacionadas, puede distorsionar los resultados (esto se llama multicolinealidad y se mide comúnmente con el VIF – Variance Inflation Factor).
3. Regresión Polinómica
Descripción: Usada cuando la relación entre la variable dependiente y la(s) variable(s) independiente(s) no es lineal, es decir, tiene una forma curva. La regresión polinómica añade términos de mayor grado (por ejemplo, x^2, 𝑥^3) a la ecuación para ajustarse a esta curvatura.
Cómo identificarla:
Al graficar los datos, observas una relación no lineal entre la variable independiente y la dependiente.
Una regresión lineal simple o múltiple da un coeficiente de determinación (R²) bajo, lo cual indica que el modelo no está capturando bien la variabilidad de los datos. Al ajustar la curva mediante regresión polinómica, el R² debería aumentar significativamente.
Para aplicar regresión polinómica, aumentas el grado del modelo (por ejemplo, a 2 para una parábola) hasta que los datos se ajusten mejor sin caer en sobreajuste.
Consejos para la Identificación en Exámenes:
Graficar los datos: Antes de aplicar cualquier modelo, haz un gráfico de los datos:
Si observas una línea recta, intenta con regresión lineal simple o múltiple, dependiendo del número de variables independientes.
Si observas una curva, explora regresión polinómica.
Interpretar el coeficiente de determinación (R²):
Si el modelo lineal simple o múltiple no ajusta bien los datos (R² bajo), y observas una curvatura en la gráfica, intenta con la regresión polinómica.
Número de variables:
Solo una variable independiente: comienza con regresión lineal simple o polinómica si ves curvatura.
Dos o más variables independientes: considera regresión múltiple. Si hay indicios de relaciones no lineales, también puedes explorar la polinómica con términos cruzados o de mayor grado.

## Tecnicas para la eliminacion de variables##

1. Selección de Variables: Técnicas Manuales y Automatizadas
1. Exhaustivo (All-in)
Descripción: Consiste en incluir todas las variables en el modelo. Esta técnica es útil cuando tienes certeza de que todas las variables son significativas y no deseas realizar pruebas adicionales de selección.
Uso: Es común cuando se trabaja bajo restricciones específicas o cuando la cantidad de variables es pequeña.
Desventajas: Aumenta el riesgo de multicolinealidad y sobreajuste (overfitting) si algunas variables no tienen una relación significativa con la variable dependiente.
2. Eliminación hacia Atrás (Backward Elimination)
Descripción: Comienza con todas las variables en el modelo y las elimina progresivamente según sus p-valores. Se eliminan las variables con los p-valores más altos (aquellas con la menor significancia estadística) hasta que todas las variables en el modelo cumplan con el nivel de significación (SL), típicamente 0.05.
Pasos:
Ajusta el modelo con todas las variables.
Identifica la variable con el p-valor más alto.
Si el p-valor es mayor que el SL, elimina esa variable.
Ajusta el modelo de nuevo y repite hasta que todas las variables tengan p-valores menores a SL.
Ventajas: Sencilla y asegura que las variables que permanecen en el modelo son estadísticamente significativas.
Ejemplo en Código (similar al que has hecho con statsmodels en Python).
3. Selección hacia Adelante (Forward Selection)
Descripción: Es lo opuesto a la eliminación hacia atrás. Comienza con un modelo vacío e incorpora progresivamente las variables más significativas (con p-valores menores).
Pasos:
Ajusta todos los modelos con cada variable independiente individualmente.
Selecciona la variable con el p-valor más bajo (siempre que sea menor al SL).
Ajusta un nuevo modelo con esa variable y añade las demás, una por una, para ver si el p-valor disminuye.
Repite hasta que ninguna variable adicional mejore el modelo.
Ventajas: Útil cuando no sabes qué variables incluir. Ayuda a construir el modelo desde una base sólida.
Desventajas: Puede ser computacionalmente costoso con muchas variables.
4. Eliminación Bidireccional (Stepwise Selection)
Descripción: Combinación de eliminación hacia atrás y selección hacia adelante. El modelo permite tanto la adición como la eliminación de variables en cada paso.
Pasos:
Establece dos niveles de significación, uno para entrada (SL_ENTER) y otro para salida (SL_STAY).
Sigue los pasos de la selección hacia adelante para añadir variables con p-valores menores a SL_ENTER.
Sigue los pasos de la eliminación hacia atrás para eliminar variables que ya no cumplen el criterio SL_STAY.
Repite hasta que no haya variables para añadir ni para eliminar.
Ventajas: Combina las ventajas de ambas técnicas y permite un ajuste más flexible y óptimo.
Desventajas: Puede resultar tedioso sin automatización y es computacionalmente intensivo en grandes conjuntos de datos.
5. Comparación de Scores (Criterio de Información de Akaike - AIC)
Descripción: Selecciona el modelo que optimiza un criterio de bondad de ajuste, como el AIC o el BIC (Criterio de Información Bayesiana). Evalúa todos los modelos posibles y elige el que tenga el mejor puntaje de ajuste.
Pasos:
Calcula el AIC o BIC para cada modelo posible (cada combinación de variables).
Selecciona el modelo con el puntaje más bajo en AIC/BIC.
Ventajas: Es muy completo y selecciona el modelo con el mejor equilibrio entre ajuste y simplicidad.
Desventajas: Con 2^n - 1 combinaciones posibles, el proceso se vuelve intensivo para grandes cantidades de variables.

## EJEMPLO DE LAS TECNICAS##
1. Exhaustivo (All-in)
Incluimos todas las variables en el modelo desde el principio.
python
Copiar código
import statsmodels.api as sm

### Agregamos una columna de unos a X para el intercepto
X_all_in = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)

### Ajustamos el modelo con todas las variables
model_all_in = sm.OLS(y, X_all_in).fit()
print(model_all_in.summary())
2. Eliminación hacia Atrás (Backward Elimination)
Aquí empezamos con todas las variables y eliminamos una por una aquellas con el p-valor más alto.
Paso 1: Crear el modelo con todas las variables
python
Copiar código
X_be = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)
model_be = sm.OLS(y, X_be).fit()
print("Paso 1 - Modelo con todas las variables")
print(model_be.summary())
Paso 2: Eliminar la variable con el p-valor más alto si supera el 0.05
Observa el resumen e identifica la variable con el p-valor más alto. Supongamos que es la columna X[:, 2].
python
Copiar código
X_be = X_be[:, [0, 1, 3, 4, 5]]  ### Eliminamos la segunda columna
model_be = sm.OLS(y, X_be).fit()
print("Paso 2 - Modelo después de eliminar X[:, 2]")
print(model_be.summary())
Paso 3: Repetir si es necesario
Si aún quedan variables con p-valores mayores a 0.05, elimina la siguiente. Supongamos que ahora la columna con p-valor más alto es X[:, 3].
python
Copiar código
X_be = X_be[:, [0, 1, 3, 5]]  # Eliminamos la columna X[:, 3]
model_be = sm.OLS(y, X_be).fit()
print("Paso 3 - Modelo después de eliminar X[:, 3]")
print(model_be.summary())
Repite el proceso hasta que todas las variables que queden tengan p-valores menores a 0.05.

3. Selección hacia Adelante (Forward Selection)
Comienza con el intercepto y añade variables una por una según su p-valor.
Paso 1: Crear el modelo con solo el intercepto
python
Copiar código
X_fs = np.ones((X.shape[0], 1)).astype(int)  # Solo el intercepto
model_fs = sm.OLS(y, X_fs).fit()
print("Paso 1 - Modelo solo con el intercepto")
print(model_fs.summary())
Paso 2: Añadir la primera variable
Prueba cada variable individualmente y elige la que tenga el p-valor más bajo. Supongamos que X[:, 1] es la mejor.
python
Copiar código
X_fs = np.append(X_fs, X[:, [1]], axis=1)
model_fs = sm.OLS(y, X_fs).fit()
print("Paso 2 - Modelo con intercepto y X[:, 1]")
print(model_fs.summary())
Paso 3: Añadir la siguiente mejor variable
Agrega la variable con el siguiente p-valor más bajo de las que quedan. Supongamos que es X[:, 3].
python
Copiar código
X_fs = np.append(X_fs, X[:, [3]], axis=1)
model_fs = sm.OLS(y, X_fs).fit()
print("Paso 3 - Modelo con intercepto, X[:, 1] y X[:, 3]")
print(model_fs.summary())
Repite hasta que añadir otra variable no mejore significativamente el modelo (p-valor alto).

4. Eliminación Bidireccional (Stepwise Selection)
Añade o elimina variables en cada paso, combinando selección hacia adelante y eliminación hacia atrás.
Paso 1: Empezamos con el intercepto
python
Copiar código
X_sws = np.ones((X.shape[0], 1)).astype(int)
model_sws = sm.OLS(y, X_sws).fit()
print("Paso 1 - Modelo solo con el intercepto")
print(model_sws.summary())
Paso 2: Selección hacia adelante: Añadir la variable con el p-valor más bajo
Supongamos que X[:, 1] es la mejor.
python
Copiar código
X_sws = np.append(X_sws, X[:, [1]], axis=1)
model_sws = sm.OLS(y, X_sws).fit()
print("Paso 2 - Modelo con intercepto y X[:, 1]")
print(model_sws.summary())
Paso 3: Eliminación hacia atrás si alguna variable no cumple SL_STAY
Supón que X[:, 1] sigue siendo significativa, pero después de agregar X[:, 3], su p-valor supera SL_STAY.
python
Copiar código
X_sws = np.append(X_sws, X[:, [3]], axis=1)
model_sws = sm.OLS(y, X_sws).fit()
print("Paso 3 - Modelo con intercepto, X[:, 1] y X[:, 3]")
print(model_sws.summary())

### Si el p-valor de alguna variable sube al agregar otra, elimínala
X_sws = X_sws[:, [0, 3]]  # Eliminamos X[:, 1] si su p-valor es alto
model_sws = sm.OLS(y, X_sws).fit()
print("Paso 4 - Modelo ajustado eliminando X[:, 1]")
print(model_sws.summary())
5. Comparación de Scores (Criterio de Información de Akaike - AIC)
Prueba con cada combinación de variables y selecciona la de AIC más bajo.
Paso 1: Modelo solo con el intercepto
python
Copiar código
X_aic = np.ones((X.shape[0], 1)).astype(int)
model_aic = sm.OLS(y, X_aic).fit()
print("AIC con solo el intercepto:", model_aic.aic)
Paso 2: Modelo con intercepto y una variable (X[:, 1])
python
Copiar código
X_aic = np.append(X_aic, X[:, [1]], axis=1)
model_aic = sm.OLS(y, X_aic).fit()
print("AIC con intercepto y X[:, 1]:", model_aic.aic)
Paso 3: Modelo con intercepto y dos variables (X[:, 1], X[:, 3])
X_aic = np.append(X_aic, X[:, [3]], axis=1)
model_aic = sm.OLS(y, X_aic).fit()
print("AIC con intercepto, X[:, 1] y X[:, 3]:", model_aic.aic)
Repite este proceso probando cada combinación hasta identificar la que tiene el AIC más bajo. Puedes comparar manualmente los AIC de cada combinación para encontrar el mejor modelo.

Con estos pasos manuales, tienes una mejor visión de cómo seleccionar variables sin necesidad de usar bucles, evaluando cada modelo con las técnicas de selección explicadas.



## Regresion Lineal ##

1. Preprocesamiento de Datos
Manejo de valores faltantes: Usaste SimpleImputer con la estrategia de la mediana para imputar valores faltantes, lo cual es adecuado. Otra opción común es utilizar la media, aunque la mediana es menos sensible a valores atípicos (outliers).
Escalado de datos: Aunque en la regresión lineal simple, el escalado no siempre es necesario, en otros tipos de regresiones o cuando se trabajan múltiples variables con diferentes escalas, es importante estandarizar o normalizar los datos para mejorar la interpretabilidad y el rendimiento.
2. División del Conjunto de Datos
Dividir el conjunto en entrenamiento y prueba, como lo haces con train_test_split, es esencial para evaluar el desempeño del modelo. En general, una proporción del 80/20 o 70/30 es buena, pero el 1/3 que usaste en el segundo código también es una práctica válida.
Asegúrate de fijar random_state para poder replicar los resultados.
3. Creación y Entrenamiento del Modelo
La clase LinearRegression de sklearn es perfecta para una regresión lineal simple. Al entrenar el modelo con fit, se ajusta la recta que mejor se adapta a los datos.
Visualización de entrenamiento: El gráfico que creas para visualizar los datos de entrenamiento y la línea de regresión es excelente. Esto ayuda a verificar si el modelo se ajusta bien a los datos.
4. Predicción y Evaluación
Evaluación en el conjunto de prueba: Utilizar el conjunto de prueba para hacer predicciones y graficarlas, como en tu segundo ejercicio, es importante para verificar que el modelo generaliza bien.
Gráficos de resultados: La visualización te ayuda a comparar visualmente las predicciones frente a los valores reales. Idealmente, la línea de regresión debe acercarse a los puntos del conjunto de prueba si el modelo está bien ajustado.
5. Métricas de Evaluación
Aunque no has incluido métricas en tus códigos, considera usar:
Error Cuadrático Medio (MSE): Mide la media de los errores al cuadrado, útil para observar cuánto se desvía la predicción del valor real.
Coeficiente de Determinación (R²): Indica qué tan bien el modelo explica la variabilidad de los datos. Un valor cercano a 1 indica un buen ajuste.
6. Conceptos Estadísticos Relevantes
Outliers: La media es sensible a los valores atípicos, como comentas en el código. En presencia de outliers, considera la mediana para imputación o incluso aplicar técnicas de detección y manejo de outliers.
Dirección de la relación: La relación positiva o negativa se observa en la pendiente de la línea de regresión. Si la pendiente es positiva, la relación entre las variables es directamente proporcional, y si es negativa, es inversamente proporcional.
7. Notas sobre Imputación y Escalado
Imputación de valores: Imputar es importante en casos de datos faltantes, y, como mencionaste, existen varias estrategias (media, mediana, vecino cercano).
Estandarización y Normalización: Aunque no es obligatorio para la regresión lineal, normalizar o estandarizar puede ser beneficioso para otros tipos de análisis. Esto ayuda a comparar variables en la misma escala, facilitando la interpretación y evitando posibles sesgos en modelos más complejos.



## Regresion lineal Multiple ##
1. Requisitos de la Regresión Lineal Múltiple
Linealidad: Las relaciones entre las variables independientes y la dependiente deben ser aproximadamente lineales.
Independencia de errores: Verifica que los errores del modelo no estén correlacionados entre sí.
Ausencia de multicolinealidad: Las variables independientes no deben estar altamente correlacionadas, lo que evitará problemas de multicolinealidad. Esto se puede evaluar usando el VIF (Variance Inflation Factor).
Normalidad multivariable: Los errores deben estar distribuidos normalmente para obtener predicciones precisas y coeficientes de regresión confiables.
2. Codificación de Variables Categóricas
Convertir variables categóricas (como el estado de la empresa o actividades extracurriculares) en variables dummy o de tipo binario es crucial. Como realizaste en ambos códigos, usaste OneHotEncoder y LabelEncoder correctamente para crear representaciones numéricas de categorías.
Evita la trampa de variables ficticias: Al crear variables dummy, elimina una columna para evitar redundancia, lo cual ya aplicaste con X = X[:, 1:].
3. Entrenamiento y Predicción
Entrenar el modelo con LinearRegression es apropiado para la regresión múltiple. Con train_test_split, divides los datos en entrenamiento y prueba, garantizando que el modelo se entrene y luego se pruebe con datos no vistos.
4. Eliminación hacia Atrás con p-valores
La eliminación hacia atrás es una excelente técnica para ajustar el modelo seleccionando únicamente las variables significativas (con un p-valor menor que tu SL de 0.05). Al seguir este proceso, reduces las variables menos influyentes en el modelo y obtienes una regresión más robusta.
En tu último código, observaste que todas las variables eran significativas, por lo cual las mantuviste en el modelo.
5. Evaluación de Predicciones y Visualización
Gráfico de valores reales vs. predicciones: Tu gráfico de dispersión entre valores reales y predicciones con una línea de identidad en rojo es una buena práctica para evaluar visualmente el rendimiento del modelo. Si los puntos están cerca de la línea roja, el modelo está haciendo predicciones precisas.
Métricas adicionales: Aunque no están en el código, considera métricas como el R² y el MSE para cuantificar la precisión del modelo.
Ejemplo Adicional de VIF para Detectar Multicolinealidad
Podríamos agregar el cálculo del Variance Inflation Factor (VIF) para verificar si alguna variable independiente está altamente correlacionada con otra, afectando la precisión de los coeficientes. ¿Te gustaría que te muestre cómo hacerlo en este mismo código?
Y nuevamente, Joe, ¡me alegra que confíes en mí para tu estudio! Cualquier otra pregunta o tema que desees profundizar, aquí estoy.

## Regresion polinómica###

1. ¿Cuándo Usar la Regresión Polinómica?
Curvatura en los Datos: Cuando los datos muestran una relación no lineal (curva), la regresión polinómica ajusta mejor al incluir términos de mayor grado (como x^2, 𝑥^3, etc.).
Ejemplo Práctico: En tu conjunto de datos sobre temperatura y ventas de helados, es lógico esperar una relación curva en lugar de una recta. A medida que la temperatura aumenta, las ventas pueden aumentar rápidamente y luego estabilizarse.
2. Construcción y Visualización del Modelo
Definir el Grado del Polinomio: Empezaste con un grado bajo (2) y lo fuiste aumentando (hasta grado 6), lo cual es ideal para encontrar el ajuste óptimo. Si bien grados altos pueden mejorar el ajuste a los datos, también aumenta el riesgo de sobreajuste (overfitting).
Grilla para Visualización: Usar X_grid con más puntos (espaciados cada 0.1) es una excelente práctica para suavizar la curva y observar cómo el modelo se ajusta en intervalos más finos.
3. Afinación del Modelo con Métricas
Evaluación de R² y MSE: Como hiciste al calcular el R² y el MSE en función del grado, puedes comparar la capacidad del modelo para explicar la variabilidad de los datos y verificar si el error medio cuadrático disminuye con un grado específico. Esto ayuda a seleccionar el grado óptimo.
4. Estrategia de Selección del Grado Polinómico
Empieza con Grados Bajos: Como en tu código, se recomienda empezar con un grado bajo (como 2 o 3).
Analiza el Incremento del R²: Si el R² mejora significativamente al aumentar el grado, continúa explorando. Si la mejora es marginal, es una señal de que ya se alcanzó el grado óptimo.
Evita el Sobreajuste: A partir de cierto grado, el modelo puede ajustarse demasiado a los puntos, reflejando variaciones aleatorias en lugar de una tendencia real.
5. Predicción con el Modelo Ajustado
Predicción para Nuevos Valores: En tu caso, usaste lin_reg2.predict(poly_reg.fit_transform([[6.5]])) para predecir el valor del salario en una posición específica (nivel 6.5). Esto es útil para explorar qué tan bien el modelo se ajusta a valores específicos y hacer predicciones con términos de grado más alto.



