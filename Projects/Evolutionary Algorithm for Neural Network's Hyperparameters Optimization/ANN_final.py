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
num_neurons = 13
n_layers = 1
train_percentage = 0.70

def normalizing(df):
    max_val = df.max(axis=0) # Se obtiene el máximo de cada columna
    min_val = df.min(axis=0) # Se obtiene el mínimo de cada columna
    range = max_val - min_val # Se obtiene la diferencia de los dos
    df = (df - min_val)/(range) # Y se utiliza para normalizarlas
    df = df.astype(float) # Se asegura que los datos sean tipo float
    return df

# Carga de datos
data = pd.read_excel("datos.xlsx") # Se lee el archivo de excel con los datos formateados
resultado = data["Class"]
data.pop("Class")
data = normalizing(data)
data = data.replace(0.000000, 0.00001)
data.insert(0, "Class", resultado)

# Distribución de los datos en sets de entrenamiento y prueba
trainset = data.sample(frac=train_percentage) # Se extraen datos para el entrenamiento
testset = data.drop(trainset.index) # Y se le quitan esos mismos al dataset para crear los datos de prueba
# Conversión de los vectores de resultado a matrices de clasificación
y_train = to_categorical(trainset["Class"], num_classes=4)
y_test = to_categorical(testset["Class"], num_classes=4)
# Creación del modelo de red neuronal
network = models.Sequential()

# Declaración de la capa de entrada y la primera capa oculta
network.add(tf.keras.layers.Dense(units=num_neurons, # número de neuronas en la capa oculta
                                  activation="sigmoid", # función de activación
                                  input_shape=(24,))) # cantidad de neuronas en la capa de entrada

# Ciclo para crear las capas de neuronas intermedias
i = 0
while i < n_layers-1: # se repite el ciclo para completar el número de capas deseadas
    network.add(layers.Dense(
    units=num_neurons, # número de neuronas
    activation="sigmoid")) # funsión de activación
    i += 1

# Declaración de la capa de salidas
network.add(layers.Dense(units=4, # cantidad de neuronas de la capa de salida
                         activation="sigmoid")) # función de activación

# Compilación del modelo
network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.019652849063277245), # se define el optimizador y la tasa de aprendizaje
                loss=tf.keras.losses.CategoricalCrossentropy()) # se define la función de pérdida
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
# Entrenamiento del modelo
losses = network.fit(x=trainset[trainset.columns[1:]],
                     y=y_train,
                     validation_data=(testset[testset.columns[1:]], y_test),
                     batch_size=100, # cantidad de datos en un batch
                     callbacks=[callback], epochs=300) # número máximo de iteraciones
# Se guarda la red creada
# network = models.load_model("red_neuronal_proyecto.keras")
# Se extrae el historial de error contra iteraciones de la clase
loss_df = pd.DataFrame(losses.history)
# Se grafican las curvas de pérdida
loss_df.loc[:, ['loss', 'val_loss']].plot()
# Y se pide que se muestre la ventana en que se graficó
plt.show()

dato = data.sample(frac=546/data.shape[0])
datoPrueba = dato.drop(columns=["Class"])
# Y se predice el resultado, luego de revertir la normalización
result = network.predict(datoPrueba)
# Se usa la función numpy.argmax para retornar el número de categoría que predijo la red
Prediction = np.argmax(result, axis=1)
# Se muestran los resultados en consola
print("Resultado:")
dato.insert(0, "Predicción", Prediction)
print(dato)
# Display de la matriz de confusión
true_labels = dato["Class"]
plt.figure(figsize=(8,8))
plt.title('Confusion Matrix', size=40, weight='bold')
sns.heatmap(confusion_matrix(true_labels, Prediction), 
            annot=True, annot_kws={'size':14, 'weight':'bold'},
            fmt='d', cbar=False, cmap='RdPu', 
            xticklabels=['MoveForward', 'SlightRightTurn', 'SharpRightTurn', 'SlightLeftTurn'],
            yticklabels=['MoveForward', 'SlightRightTurn', 'SharpRightTurn', 'SlightLeftTurn'])
plt.tick_params(axis='both', labelsize=14)
plt.ylabel('Actual', size=14, weight='bold')

plt.xlabel('Predicted', size=14, weight='bold')
plt.show()
# Display del reporte de clasificación
print("\n-----------------------------------------------------")
print("Classification report:\n")
print(classification_report(true_labels, Prediction, digits=4,
target_names=['MoveForward', 'SlightRightTurn','SharpRightTurn','SlightLeftTurn']))
print("-----------------------------------------------------")