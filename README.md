# Ventas-red-neuronal
Es un algoritmo de red neuronal de 1 capa  simple para  predecir resultados  de ventas  de un local de helados en funcion a la temperatura.
Descripcion del Algoritmo.
Importar las bibliotecas necesarias: TensorFlow, pandas, NumPy, seaborn y matplotlib.
Leer los datos de ventas desde un archivo CSV llamado "ventas.csv" y guardarlos en un DataFrame de pandas llamado ventas_df.
Visualizar los datos mediante un gráfico de dispersión, donde la temperatura se representa en el eje x y las ventas en el eje y.
Crear el conjunto de entrenamiento asignando la columna "Temperatura (°C)" a X_train y la columna "Ventas en Dólares ($)" a y_train.
Crear un modelo secuencial utilizando la API de Keras en TensorFlow. Este modelo consiste en una capa densa (totalmente conectada) con una unidad. La capa toma como entrada una forma de [1], que corresponde a la temperatura.
Imprimir un resumen del modelo, que proporciona información sobre la arquitectura y el número de parámetros.
Compilar el modelo especificando el optimizador (Adam con una tasa de aprendizaje de 0.1) y la función de pérdida (error cuadrático medio).
Entrenar el modelo utilizando el conjunto de entrenamiento (X_train e y_train) durante un número determinado de épocas (en este caso, 1000).
Almacenar el historial del proceso de entrenamiento en epochs_hist, que contiene información sobre el valor de pérdida en cada época.
Acceder a las claves de epochs_hist.history, que representan las métricas del historial de entrenamiento, como la pérdida.
Obtener los pesos del modelo entrenado utilizando model.get_weights() y almacenarlos en la variable weights.
Este código demuestra una implementación sencilla de un modelo de machine learning utilizando TensorFlow para predecir las ventas en función de la temperatura. El modelo se entrena utilizando el conjunto de entrenamiento y la función de pérdida de error cuadrático medio.
