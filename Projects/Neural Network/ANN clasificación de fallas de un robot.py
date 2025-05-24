# Importación de librerías
import pandas as pd
import tensorflow as tf
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score


# Definición de hiperparámetros
num_neurons = 15
n_layers = 1
train_percentage = 0.80

# Carga de datos
data = pd.read_csv("All_data_1.csv") # Se lee el archivo de excel con los datos formateados
print(data)
data.pop("Unnamed: 0") # Elimina una columna innecesaria
print(data)
# Distribución de los datos en sets de entrenamiento y prueba
trainset = data.sample(frac=train_percentage) # Se extraen datos para el entrenamiento
testset = data.drop(trainset.index) # Y se le quitan esos mismos al dataset para crear los datos de prueba

# Conversión de los vectores de resultado a matrices de clasificación
y_train = to_categorical(trainset["Resultado"], num_classes=4)
y_test = to_categorical(testset["Resultado"], num_classes=4)
# Creación del modelo de red neuronal
network = models.Sequential()

# Declaración de la capa de entrada y la primera capa oculta
network.add(tf.keras.layers.Dense(units=num_neurons, # número de neuronas en la capa oculta
                                  activation="sigmoid", # función de activación
                                  input_shape=(90,))) # cantidad de neuronas en la capa de entrada
# Ciclo para crear las capas de neuronas intermedias
i = 0
while i < n_layers-1: # se repite el ciclo para completar el número de capas deseadas
    network.add(layers.Dense( units=num_neurons, # número de neuronas
                             activation="sigmoid")) # funsión de activación
    i += 1
# Declaración de la capa de salidas
network.add(layers.Dense(
    units=4, # cantidad de neuronas de la capa de salida
    activation="sigmoid")) # función de activación

# Compilación del modelo
network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), # se define el optimizador y la tasa de aprendizaje
                loss=tf.keras.losses.CategoricalCrossentropy()) # se define la función de pérdida

# Entrenamiento del modelo
losses = network.fit(x=trainset[trainset.columns[1:]], y=y_train,
                     validation_data=(testset[testset.columns[1:]], y_test),
                     batch_size=100, # cantidad de datos en un batch
                     epochs=600) # número máximo de iteraciones
# Se guarda la red creada
network.save("red_neuronal_1.keras")
# Se extrae el historial de error contra iteraciones de la clase
loss_df = pd.DataFrame(losses.history)
# Se grafican las curvas de pérdida
loss_df.loc[:, ['loss', 'val_loss']].plot()
# Y se pide que se muestre la ventana en que se graficó
plt.show()

# Se eligen 15 datos al azar
dato = data.sample(frac=15/data.shape[0])
datoPrueba = dato.drop(columns=["Resultado"])
# Y se predice el resultado, luego de revertir la normalización
result = network.predict(datoPrueba)
# Se usa la función numpy.argmax para retornar el número de categoría que fue predicho
Prediction = np.argmax(result, axis=1)
# Se muestran los resultados en consola
print("Resultado:")
dato.insert(0, "Predicción", Prediction)
print(dato)
# Display de la matriz de confusión
true_labels = dato["Resultado"]
plt.figure(figsize=(4,4))
plt.title('Confusion Matrix', size=20, weight='bold')
sns.heatmap(confusion_matrix(true_labels, Prediction), annot=True, 
            annot_kws={'size':14, 'weight':'bold'}, fmt='d', cbar=False, cmap='RdPu',
            xticklabels=['fr_collision', 'normal', 'obstruction', 'collision'],
            yticklabels=['fr_collision', 'normal', 'obstruction', 'collision'])
plt.tick_params(axis='both', labelsize=14)
plt.ylabel('Actual', size=14, weight='bold')
plt.xlabel('Predicted', size=14, weight='bold')
plt.show()

# Display del reporte de clasificación
print("\n-----------------------------------------------------")
print("Classification report:\n")
print(classification_report(true_labels, Prediction, digits=4,
target_names=['fr_collision', 'normal', 'obstruction', 'collision']))
print("-----------------------------------------------------")
