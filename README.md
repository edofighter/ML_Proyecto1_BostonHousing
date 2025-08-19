# Proyecto ML1: Predicción de Precios de Viviendas (Boston Housing)

Este proyecto implementa un **modelo de regresión lineal** en Python para predecir los precios de las viviendas del dataset de Boston. Incluye **exploración de datos (EDA), entrenamiento, predicciones, evaluación y visualización** de resultados.

---

## 📂 Estructura del Proyecto
Proyecto_ML1/
│
├─ ml1.py # Script principal con todo el flujo de ML
├─ README.md # Descripción del proyecto
├─ requirements.txt # Librerías necesarias
└─ .gitignore # Archivos a ignorar


---

## 🛠 Librerías utilizadas

- `pandas` – manejo de datos
- `numpy` – operaciones numéricas
- `matplotlib` y `seaborn` – visualización de datos
- `scikit-learn` – modelado de regresión lineal y métricas

---

## 🔹 Flujo del proyecto

1. **Carga del dataset** desde OpenML (`Boston Housing`).
2. **Exploración de datos (EDA)**:
   - Estadísticas descriptivas
   - Detección de valores nulos
   - Matriz de correlación
   - Relación entre número de habitaciones (`RM`) y precio
3. **División de los datos** en entrenamiento y prueba (70%-30%).
4. **Entrenamiento del modelo** de regresión lineal.
5. **Predicciones** con el conjunto de prueba.
6. **Evaluación del modelo** con métricas:
   - MAE (Error Absoluto Medio)
   - MSE (Error Cuadrático Medio)
   - RMSE (Raíz del Error Cuadrático Medio)
   - R² (Coeficiente de determinación)
7. **Visualización de resultados**:
   - Scatter plot de valores reales vs predichos
   - Histograma de errores de predicción

---

## 📊 Ejemplo de métricas obtenidas

| Métrica | Valor |
|---------|-------|
| MAE     | XX.XX |
| MSE     | XX.XX |
| RMSE    | XX.XX |
| R²      | XX.XX |

> Los valores exactos dependen de la ejecución del modelo con tus datos.

---

## 📈 Visualizaciones

1. **Comparación real vs predicción**  

2. **Distribución de errores**  
 
*(Opcional: guardar las imágenes con `plt.savefig("nombre.png")` en tu script.)*

---

