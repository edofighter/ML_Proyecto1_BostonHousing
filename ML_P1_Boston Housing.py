#libreruias Basicas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

#librerias de Machine Learning
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#Cargar y explorar el dataset
#dataset de viviendas de Boston
boston = fetch_openml(name = 'boston', version = 1, as_frame = True)

#Extraemos X (features) y y (target) 
X = boston.data
Y = pd.to_numeric(boston.target) #directamente numérico

#Convertir la variable objetivo a numérica
y = pd.to_numeric(Y)

#Mostar dimensiones
print('Dimensiones de X:', X.shape)
print("Dimensiones de y:", Y.shape if hasattr(Y, "shape") else (len(Y),))

# Mostrar primeras filas
print('\nPrimeras filas del dataset')
print(X.head())

print('\nVariable objetivo (precio de la vivienda):')
print(Y.head())


#Exploración de Datos (EDA)
#Estadísticas básicas de las variables
print('\nEstadistica descriptivas:')
print(X.describe())

#existencia de  valores nulos
print('\nValores nulos en el dataset:')
print(X.isnull().sum())

# Historial variable objetivo (precio)
plt.figure(figsize =(6,4))
sns.histplot(Y, kde = True, bins = 30 )
plt.title('Distribución de los precios de viviendas')
plt.xlabel('Precio (en miles de USD)')
plt.ylabel('Frecuencia')
#plt.show()

#Matriz de correlación
plt.figure(figsize = (12,8))
sns.heatmap(X.corr(), annot = True, cmap = 'coolwarm')
plt.title('Matriz de correlación de las variables')
#plt.show()

#Relación entre número de habitaciones (RM) y precio
plt.figure(figsize = (6,4))
sns.scatterplot(x = X['RM'],y = Y)
plt.title('Relación entre número de habitaciones y precio')
plt.xlabel('Número de habitaciones (RM)')
plt.ylabel('Precio (en miles de USD)')
#plt.show()


#Dividir los datos en entrenamiento y prueba
#Dividir datos (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print('Tamaño de X_train:', X_train.shape)
print('Tamaño de X_test:', X_test.shape)
print('Tamaño de y_train:', y_train.shape)
print('Tamaño de y_test:', y_test.shape)


#Entrenar el modelo de regresión lineal
# Convertir a NumPy arrays, conservando todas las columnas
X_train = np.array(X_train, dtype=float)
X_test  = np.array(X_test, dtype=float)
y_train = np.array(y_train, dtype=float)
y_test  = np.array(y_test, dtype=float)

#crear modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Mostrar los coeficientes del modelo
print('Intercepto (β0):', modelo.intercept_)
print('Pendiente (β1):', modelo.coef_[0])

#predicciones con el conjunto de prueba
Y_pred = modelo.predict(X_test)

# Mostrar algunas comparaciones reales vs predichas
for real, pred in zip(y_test[:10], Y_pred[:10]):
    print(f'Real: {real:.2f} | Predicción: {pred:.2f}')


#Evaluación del modelo
#carcular metricas
mae = mean_absolute_error(y_test, Y_pred)
mse = mean_squared_error(y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, Y_pred)

#mostar resultados
print(f'MAE  (Error absoluto medio): {mae:.2f}')
print(f'MSE  (Error cuadrático medio): {mse:.2f}')
print(f'RMSE (Raíz del error cuadrático medio): {rmse:.2f}')
print(f'R²   (Coeficiente de determinación): {r2:.2f}')


#Visualización y conclusiones
#Comparación real vs predicho
plt.figure(figsize = (8,6))
sns.scatterplot(x = y_test, y = Y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valores reales')
plt.ylabel('Valores predichos')
plt.title('Comparación: Valores Reales vs Predichos')
plt.show()

# Distribución de errores
errores = y_test - Y_pred
plt.figure(figsize=(8,6))
sns.histplot(errores, kde=True, bins=30)
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.title("Distribución de los errores de predicción")
plt.show()