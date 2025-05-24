from deap import base, creator, tools, algorithms
import random
import numpy
import array
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
train_percentage = 0.75
iteraciones = 300
LABELS = ['MoveForward', 'SlightRightTurn', 'SharpRightTurn','SlightLeftTurn']

def customMutation(individuo, indpb=0.2):
    # Diccionario que contiene los límites de los espacios de alelos
    lim_dict = {0: [1, 6], 1: [1, 14], 2: [0.00001, 0.5], 3: [1, 3], 4: [0,1]}
    # Se itera a través de cada uno de lo genes
    for index, gen in enumerate(individuo):
        prob = random.random()
        # Si el valor generado con anterioridad es menor a la probabilidad
        # definida por el usuario, se cambia (muta) el gen.
        if prob < indpb:
            new_gen = gen
            # Se repite hasta que se encuentra un gen de valor diferente
            while new_gen == gen:
                # Todos los genes son enteros con excepción del tercero (index 2)
                if index != 2:
                    new_gen = random.randint(lim_dict[index][0],
                    lim_dict[index][1])
                else:
                    new_gen = random.uniform(lim_dict[index][0],
                    lim_dict[index][1])
                # Se asigna como float porque todos los genes son tipo float
                individuo[index] = float(new_gen)
    # Se retorna el individuo mutado
    return individuo

def normalizing(df):
    max_val = df.max(axis=0) # Se obtiene el máximo de cada columna
    min_val = df.min(axis=0) # Se obtiene el mínimo de cada columna
    range = max_val - min_val # Se obtiene la diferencia de los dos
    df = (df - min_val)/(range) # Y se utiliza para normalizarlas
    df = df.astype(float) # Se asegura que los datos sean tipo float
    return df

# Carga de datos
def load_data():
    data = pd.read_excel("datos.xlsx") # Se lee el archivo de excel con los datos formateados
    resultado = data["Class"]
    data.pop("Class")
    data = normalizing(data)
    data = data.replace(0.000000, 0.00001)
    data.insert(0, "Class", resultado)
    return data

def Report(dato, Prediction):
    true_labels = dato["Class"]
    report_final = classification_report(true_labels, Prediction, target_names=LABELS, output_dict=True, zero_division=0)
    precision = []
    for label in LABELS:
        precision.append(report_final[label]["precision"])
    return precision

def define_model(network, n_layers, number_neurons, LR, optimizer, Momentum):
    # Declaración de la capa de entrada y la primera capa oculta
    network.add(tf.keras.layers.Dense(units=number_neurons, # número de neuronas en la capa oculta
                                      activation="sigmoid", # función de activación
                                      input_shape=(24,))) # cantidad de neuronas en la capa de entrada
    # Ciclo para crear las capas de neuronas intermedias
    i = 0
    while i < n_layers-1: # se repite el ciclo para completar el número de capas deseadas
        network.add(layers.Dense(units=number_neurons, # número de neuronas
                                 activation="sigmoid")) # funsión de activación
        i += 1
    # Declaración de la capa de salidas
    network.add(layers.Dense(units=4, # cantidad de neuronas de la capa de salida
                             activation="sigmoid")) # función de activación
    if Momentum == 1:
        Momentum = 0.90
    # Compilación del modelo
    if optimizer == 1:
        # Cambiar por RMSprop
        network.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR), # se define el optimizador y la tasa de aprendizaje
        loss=tf.keras.losses.CategoricalCrossentropy()) # se define la función de pérdida
    elif optimizer == 2:
        network.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LR, momentum=Momentum), # se define el optimizador y la tasa de aprendizaje
                        loss=tf.keras.losses.CategoricalCrossentropy()) # se define la función de pérdida
    elif optimizer == 3:
        network.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=LR, momentum=Momentum), # se define el optimizador y la tasa de aprendizaje
                        loss=tf.keras.losses.CategoricalCrossentropy()) # se define la función de pérdida

def red_neuronal(num_neurons, n_layers, learning_rate, optimizador, Momentum, data):
    # Distribución de los datos en sets de entrenamiento y prueba
    trainset = data.sample(frac=train_percentage) # Se extraen datos para el entrenamiento
    testset = data.drop(trainset.index) # Y se le quitan esos mismos al dataset para crear los datos de prueba
    # Conversión de los vectores de resultado a matrices de clasificación
    y_train = to_categorical(trainset["Class"], num_classes=4)
    y_test = to_categorical(testset["Class"], num_classes=4)
    # Creación del modelo de red neuronal
    network = models.Sequential()
    define_model(network, n_layers=n_layers, number_neurons=num_neurons,
                 LR=learning_rate, optimizer=optimizador, Momentum=Momentum)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    # Entrenamiento del modelo
    losses = network.fit(x=trainset[trainset.columns[1:]], y=y_train,
                         validation_data=(testset[testset.columns[1:]], y_test),
                         batch_size=100, # cantidad de datos en un batch
                         callbacks=[callback], epochs=iteraciones,
                         verbose=0) # número máximo de iteraciones
    # Se extrae el historial de error contra iteraciones de la clase
    loss_df = pd.DataFrame(losses.history)
    epochs = len(loss_df)
    loss = loss_df.loc[epochs-1, "loss"]
    val_loss = loss_df.loc[epochs-1, "val_loss"]

    # Se eligen 15 datos al azar
    dato = data.sample(frac=1)
    datoPrueba = dato.drop(columns=["Class"])
    # Y se predice el resultado, luego de revertir la normalización
    result = network.predict(datoPrueba)
    # Se usa la función numpy.argmax para retornar el número de categoría que predijo la red
    Prediction = np.argmax(result, axis=1)
    # Se sacan las precisiones para cada una de las categorías
    precision = Report(dato, Prediction)
    # El return depende de lo que se necesite para la función de calidad
    return precision, loss, val_loss

def funcionEval(individuo, data):
    Ncapas = individuo[0]
    Nneuronas = individuo[1]
    Learning_rate = individuo[2]
    Optimizador = individuo[3]
    Momentum = individuo[4]
    # Acá es donde se debe de llamar a la red neuronal y manipular los valores que devuelve para darle una puntuación a cada una de las posibles soluciones.
    if (Optimizador == 1) and (Momentum == 0):
        error = 10
        return [error, ]
    else:
        avg_precision, loss, val_loss = red_neuronal(Nneuronas, Ncapas, Learning_rate, Optimizador, Momentum, data)
        sum_precision = sum(avg_precision)
        error = loss + val_loss + (4 - sum_precision)
        print(Ncapas, Nneuronas, Learning_rate, Optimizador, Momentum, error)
        return [error, ]
    
def AG():
    data = load_data()
    generaciones = 10
    tamañoPoblación = 100
    mutation_gen = 0.1
    # Definición del toolbox
    toolbox = base.Toolbox()
    # El fitness que manejará el toolbox será una función con el peso 1
    # (maximización con peso unitario para cada atributo)
    creator.create('Fitness_funct', base.Fitness, weights=(-1.0,))
    # El individuo que manejará el toolbox será un array de floats
    creator.create('Individuo', array.array, fitness=creator.Fitness_funct, typecode='f')

    # Registro de la función de evaluación, usando lo definido previamente en el código
    toolbox.register('evaluate', funcionEval, data=data)
    # Acá es donde hay que definir distintos atributos con diferentes rangos:
    toolbox.register('capas', random.randint, a=1, b=6)
    toolbox.register('neuronas', random.randint, a=1, b=14)
    toolbox.register('LR', random.uniform, a=0.00001, b=0.5)
    toolbox.register('optimizador', random.randint, a=1, b=3)
    toolbox.register('momento', random.randint, a=0, b=1)
    # Se genera un atributo de float al azar 3 veces, y se guarda en un Individuo
    toolbox.register('individuo_gen', tools.initCycle, creator.Individuo,
                     (toolbox.capas, toolbox.neuronas, toolbox.LR, toolbox.optimizador,
                      toolbox.momento), n=1)
    toolbox.register("Poblacion", tools.initRepeat, list,
    toolbox.individuo_gen, n=tamañoPoblación)
    # Para ello, llama unas 30 veces a la función 'individuo_gen', de manera que
    # queda generada una población de 'Individuo's.
    # Se utiliza la función registrada para generar una población
    popu = toolbox.Poblacion()
    print(popu)

    # Método de cruce de dos puntos
    toolbox.register('mate', tools.cxOnePoint)
    # Para la mutación, se utiliza el método de mutación Gaussiana
    toolbox.register('mutate', customMutation, indpb=0.2)
    # Para la mutación, se utiliza el método de torneo
    toolbox.register('select', tools.selTournament, tournsize=2)
    # Hall of Fame: presentación de los mejores 10 individuos
    hof = tools.HallOfFame(10)
    # Estadísticas del fitness general de la población
    stats = tools.Statistics(lambda indiv: indiv.fitness.values)
    stats.register('avg', numpy.mean) # Promedio de la gen
    stats.register('std', numpy.std) # Desviación estándar de los individuos
    stats.register('min', numpy.min) # Fitness mínimo de la gen
    stats.register('max', numpy.max) # Fitness máximo de la gen

    # Una vez que todo está registrado y establecido, ya se puede comenzar
    # a correr el algoritmo evolutivo.
    popu, logbook = algorithms.eaSimple(popu, toolbox, cxpb=0.45, mutpb=mutation_gen,
                                        ngen=generaciones, stats=stats, halloffame=hof,
                                         verbose=True)
    print('---------------------------')
    print(logbook)
    print(hof)
    # X axis parameter:
    generation = logbook.select("gen")
    # Y axis parameter:
    avg = logbook.select("avg")
    std = logbook.select("std")
    minn = logbook.select("min")
    plt.plot(generation, avg, label="Calidad promedio")
    plt.plot(generation, std, label="Desviación estándar")
    plt.plot(generation, minn, label="Mejor individuo")
    plt.xlabel('Generación')
    plt.ylabel('Valor')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    AG()




