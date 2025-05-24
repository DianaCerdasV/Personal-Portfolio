# Importación de librerías
import pandas as pd
import tensorflow as tf
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from mpl_toolkits.mplot3d import Axes3D

# Definición de hiperparámetros
num_neurons = 3
n_layers = 3
train_percentage = 0.70


file_path = "C:\\Users\\diana\\UtilityB.csv"
# Intentar cargar datos existentes si el archivo ya existe
# Intentar cargar datos existentes si el archivo ya existe
try:
    df = pd.read_csv(file_path)
    data = df.values.tolist()
    id = df['id'].max() +1
except FileNotFoundError:
    data = []
    id = 1

# Definir los valores mínimo y máximo
min_val = 0
max_val = np.sqrt(8000000)

# Normalizar solo las columnas de distancia
df["Red Distance"] = (df["Red Distance"] - min_val) / (max_val - min_val)
df["Green Distance"] = (df["Green Distance"] - min_val) / (max_val - min_val)
df["Blue Distance"] = (df["Blue Distance"] - min_val) / (max_val - min_val)
# Crear la gráfica
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# Crear la gráfica
x = df["Red Distance"]   # Eje X: Distancia al objeto rojo
y = df["Green Distance"] # Eje Y: Distancia al objeto verde
z = df["Blue Distance"]  # Eje Z: Distancia al objeto azul
score = df["Score"]      # Color basado en el puntaje
sc = ax.scatter(x, y, z, c=score, cmap="viridis", edgecolors="k", s=100)
# Crear la gráfica de dispersión 3D


# Agregar la barra de colores
plt.colorbar(sc, label="Score")

# Etiquetas y título
ax.set_xlabel("Distancia a objeto rojo")
ax.set_ylabel("Distancia a objeto verde")
ax.set_zlabel("Distancia a objeto azul")
ax.set_title("Gráfico 3D de distancias y Score")

# Mostrar la gráfica
plt.show()



# Distribución de los datos en sets de entrenamiento y prueba
trainset = df.sample(frac=train_percentage) # Se extraen datos para el entrenamiento
tempSet = df.drop(trainset.index)
testset = df.sample(frac=0.7)
tempSet = df.drop(testset.index)
print(testset)

x_train = trainset.drop(columns=["Score"])
x_test = testset.drop(columns=["Score"])
y_train = trainset["Score"]
y_test = testset["Score"]


# Creación del modelo de red neuronal
network = models.Sequential()

# Declaración de la capa de entrada y la primera capa oculta
network.add(tf.keras.layers.Dense(units=num_neurons, # número de neuronas en la capa oculta
                                  activation="sigmoid", # función de activación
                                  input_shape=(3,))) # cantidad de neuronas en la capa de entrada

# Ciclo para crear las capas de neuronas intermedias
i = 0
while i < n_layers-1: # se repite el ciclo para completar el número de capas deseadas
    network.add(layers.Dense(units=num_neurons, # número de neuronas
                             activation="sigmoid")) # función de activación
    i += 1

# Declaración de la capa de salidas
network.add(layers.Dense(units=1, # cantidad de neuronas de la capa de salida
                         activation="linear")) # función de activación

# Compilación del modelo
network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), # se define el optimizador y la tasa de aprendizaje
loss=tf.keras.losses.MSE) # se define la función de pérdida
# Entrenamiento del modelo
#print(x_train[x_train.columns[1:]])
losses = network.fit(x=x_train[x_train.columns[1:]], 
                     y=y_train,validation_data=(x_test[x_test.columns[1:]],y_test),
                     batch_size=10, # cantidad de datos en un batch
                     epochs=70) # número máximo de iteraciones
# Se guarda la red creada
network.save("C:\\Users\\diana\\ANNB.keras")
# Se extrae el historial de error contra iteraciones de la clase
loss_df = pd.DataFrame(losses.history)
# Se grafican las curvas de pérdida
loss_df.loc[:, ['loss', 'val_loss']].plot()
# Y se pide que se muestre la ventana en que se graficó
plt.show()

dato = tempSet
resultado_real = dato["Score"]
datoPrueba = dato.drop(columns=["Score"])
# Y se predice el resultado, luego de revertir la normalización
result = network.predict(datoPrueba[datoPrueba.columns[1:]])
# Se usa la función numpy.argmax para retornar el número de categoría que fue predicha
Prediction = np.argmax(result, axis=1)
# Se muestran los resultados en consola
print("Resultado:")
dato.insert(0, "Predicción", result)
print(dato)
# Display de la matriz de confusión
true_labels = dato["Score"]


# Graficar valores reales vs predicciones
plt.scatter(resultado_real, result, alpha=0.5)
plt.plot([min(resultado_real), max(resultado_real)], [min(resultado_real), max(resultado_real)], '--r')  # Línea de referencia (y=x)
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Comparación de Predicciones vs Reales")
plt.show()
error = resultado_real - result.flatten()  # Diferencia entre real y predicho

plt.hist(error, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel("Error (Real - Predicho)")
plt.ylabel("Frecuencia")
plt.title("Distribución del Error")
plt.show()

""""
plt.figure(figsize=(5,5))
plt.title('Confusion Matrix', size=20, weight='bold')
sns.heatmap(
confusion_matrix(true_labels, Prediction), annot=True, 
annot_kws={'size':14, 'weight':'bold'}, fmt='d', cbar=False,
cmap='RdPu', xticklabels=['-90', '-45', '0', '45', '90'],
yticklabels=['-90', '-45', '0', '45', '90'])
plt.tick_params(axis='both', labelsize=14)
plt.ylabel('Actual', size=14, weight='bold')
plt.xlabel('Predicted', size=14, weight='bold')
plt.show()
# Display del reporte de clasificación
print("\n-----------------------------------------------------")
print("Classification report:\n")
print(classification_report(true_labels, Prediction, digits=5,
target_names=['-90', '-45', '0', '45', '90']))
print("-----------------------------------------------------")"
"""