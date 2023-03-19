import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Importando Datos
ventas_df = pd.read_csv("ventas.csv")

#Visualizacion de Datos 
sns.scatterplot(ventas_df['Temperatura (°C)'], ventas_df['Ventas en Dólares ($)'])

#Creando set de entrenamiento
X_train = ventas_df['Temperatura (°C)']
y_train = ventas_df['Ventas en Dólares ($)']

#Creando el Modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

model.summary()

#Compilado
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

#Entrenando el modelo
epochs_hist = model.fit(X_train, y_train, epochs = 1000)

#Evaluando el modelo
keys = epochs_hist.history.keys()


weights = model.get_weights()
#print(weights)





