# Importación de librerías
from tensorflow.keras.models import load_model

import pandas as pd
import tensorflow as tf
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
val_data =[]


for i in np.arange(0, 0.8, 0.05):
    for n in np.arange(0, 0.8, 0.05):
        for k in np.arange(0, 0.8, 0.05):
            val_data.append([i, n, k])
            df = pd.DataFrame(val_data, columns=["Red Distance", "Green Distance", "Blue Distance"])
        
def evaluarPred(candidates):
    evaluated_candidates = []
    for i in range(len(candidates)):
        candidatesP = candidates[i][:]
        candidatesP.pop(2)
        #print("Candidatos", candidates[i], candidatesP)
        valuation = result = network.predict(candidates)
        evaluated_candidates.append((candidates[i],) + (valuation,))
        #print("Candidatos:", evaluated_candidates)
        #evaluated_candidates = candidates[i] + [valuation]

    # Ordenor los estados evaluados
    evaluated_candidates.sort(key=lambda x: x[-1])

    return evaluated_candidates

network = load_model("C:\\Users\\diana\\ANNRed.keras")



resultR = network.predict(df)

network2 = load_model("C:\\Users\\diana\\ANNG.keras")
resultG = network2.predict(df)
network3 = load_model("C:\\Users\\diana\\ANNB.keras")
resultB = network3.predict(df)


figR = plt.figure(figsize=(8, 6))
# Crear la gráfica
ax1 = figR.add_subplot(111, projection='3d')

# Crear la gráfica
y = df["Red Distance"]   # Eje X: Distancia al objeto rojo
x = df["Green Distance"] # Eje Y: Distancia al objeto verde
z = df["Blue Distance"]  # Eje Z: Distancia al objeto azul
score = resultR      # Color basado en el puntaje
sc = ax1.scatter(x, y, z, c=score, cmap="viridis", edgecolors="k", s=100)
# Crear la gráfica de dispersión 3D
"""
# Ajustar los límites para que (0,0,0) esté en la esquina inferior izquierda
ax.set_xlim([0, max(x)])  # Eje X inicia en 0
ax.set_ylim([0, max(y)])  # Eje Y inicia en 0
ax.set_zlim([0, max(z)])  # Eje Z inicia en 0"""
# Agregar la barra de colores
plt.colorbar(sc, label="Score")

# Etiquetas y título
ax1.set_xlabel("Distancia a objeto verde")
ax1.set_ylabel("Distancia a objeto rojo")
ax1.set_zlabel("Distancia a objeto azul")
ax1.set_title("Modelo Rojo")

figG = plt.figure(figsize=(8, 6))
# Crear la gráfica
ax2 = figG.add_subplot(111, projection='3d')

# Crear la gráfica
x3 = df["Red Distance"]   # Eje X: Distancia al objeto rojo
y3 = df["Green Distance"] # Eje Y: Distancia al objeto verde
z3 = df["Blue Distance"]  # Eje Z: Distancia al objeto azul
scoreG = resultG      # Color basado en el puntaje
sc1 = ax2.scatter(x3, y3, z3, c=scoreG, cmap="viridis", edgecolors="k", s=100)
# Crear la gráfica de dispersión 3D
"""
# Ajustar los límites para que (0,0,0) esté en la esquina inferior izquierda
ax.set_xlim([0, max(x)])  # Eje X inicia en 0
ax.set_ylim([0, max(y)])  # Eje Y inicia en 0
ax.set_zlim([0, max(z)])  # Eje Z inicia en 0"""
# Agregar la barra de colores
plt.colorbar(sc1, label="Score")

# Etiquetas y título
ax2.set_xlabel("Distancia a objeto rojo")
ax2.set_ylabel("Distancia a objeto verde")
ax2.set_zlabel("Distancia a objeto azul")
ax2.set_title("Modelo Verde")

figB = plt.figure(figsize=(8, 6))
# Crear la gráfica
ax = figB.add_subplot(111, projection='3d')

# Crear la gráfica
x2 = df["Red Distance"]   # Eje X: Distancia al objeto rojo
z2 = df["Green Distance"] # Eje Y: Distancia al objeto verde
y2 = df["Blue Distance"]  # Eje Z: Distancia al objeto azul
scoreB = resultB      # Color basado en el puntaje
sc2 = ax.scatter(x2, y2, z2, c=scoreB, cmap="viridis", edgecolors="k", s=100)
# Crear la gráfica de dispersión 3D
"""
# Ajustar los límites para que (0,0,0) esté en la esquina inferior izquierda
ax.set_xlim([0, max(x)])  # Eje X inicia en 0
ax.set_ylim([0, max(y)])  # Eje Y inicia en 0
ax.set_zlim([0, max(z)])  # Eje Z inicia en 0"""
# Agregar la barra de colores
plt.colorbar(sc2, label="Score")

# Etiquetas y título
ax.set_xlabel("Distancia a objeto rojo")
ax.set_ylabel("Distancia a objeto azul")
ax.set_zlabel("Distancia a objeto verde")
ax.set_title("Modelo Azul")
# Mostrar la gráfica
plt.show()