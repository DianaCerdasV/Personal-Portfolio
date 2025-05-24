# Importación de librerías
import pandas as pd
from keras import models
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Variables globales con el número de errores
errores = 0 # errores de la red original
errores_aux = 0 # errores de la red modificada
# Se carga el modelo ya definido anteriormente
network = models.load_model("red_neuronal_1.keras")
df = pd.read_csv("All_data.csv") # Lee el archivo de excel original
df.pop("Unnamed: 0") # Elimina una columna innecesaria
true_labels = df["Resultado"] # Guarda los resultados correctos
datoPrueba = df.drop(columns=["Resultado"]) # Genera un df solo con entradas
result = network.predict(datoPrueba) # se predicen los resultados con el dataframe de prueba
Prediction = np.argmax(result, axis=1) # se usa la función numpy.argmax para extraer el número de clase predicho
# Se genera la matriz de confusión
Matrix = confusion_matrix(true_labels, Prediction).ravel()
# Descomentar el siguiente código si se quiere mostrar la matriz de confusión
""""
plt.figure(figsize=(4,4))
plt.title('Confusion Matrix', size=20, weight='bold')
sns.heatmap(
confusion_matrix(true_labels, Prediction),
annot=True,
annot_kws={'size':14, 'weight':'bold'},
fmt='d',
cbar=False,
cmap='RdPu',
xticklabels=['fr_collision', 'normal', 'obstruction', 'collision'],
yticklabels=['fr_collision', 'normal', 'obstruction', 'collision'])
plt.tick_params(axis='both', labelsize=14)
plt.ylabel('Actual', size=14, weight='bold')
plt.xlabel('Predicted', size=14, weight='bold')
plt.show()
"""

# Ciclo para contar el número de errores en la clasificación de los datos
for a in range(len(Matrix)):
    if a not in [0 , 5, 10, 15]:
        errores = errores + Matrix[a] # se suman todos los datos de la matriz que no estén en la diagonal (número de clasificaciones incorrectas)

# Variables necesarias para el estudio de sensibilidad
df2 = pd.DataFrame([0], columns=["10%"], index=["Original"]) # Dataframe para guardar el resultado
col_names = ["10%", "15%", "20%", "25%"] # Nombres de las columnas
functions_list = [lambda x:x*1.10, lambda x:x*1.15, lambda x:x*1.20, lambda
x:x*1.25] # lista de funciones anónimas usadas para escalar los datos del dataframe


# Ciclo para realizar el estudio de sensibilidad
for j in range(len(functions_list)):
    for i in range(1, len(df.columns)):
        df_aux = df.copy() # Copia el dataframe original
        # Las próximas tres lineas cambian la columna correspondiente a los valores modificados
        col = df[df.columns[i]]
        change_col = col.apply(functions_list[j])
        df_aux[df.columns[i]] = change_col
        entrada = df_aux.drop(columns=["Resultado"]) # Genera los valores de entrada a la red
        result_aux = network.predict(entrada) # Hace la predicción
        # Pasa a binario lo que predijo la red
        Prediction_aux = np.argmax(result_aux, axis=1)
        # Genera los valores correctos o de error
        Matrix_aux = confusion_matrix(true_labels, Prediction_aux).ravel()
        # Calcula la cantidad total de errores
        for b in range(len(Matrix_aux)):
            if b not in [0 , 5, 10, 15]:
                errores_aux = errores_aux + Matrix_aux[b]
        # Saca la diferencia de errores entre los valores originales y con la tabla modificada
        diferencia = errores - errores_aux # Si es positivo quiere decir que mejoró la red
        errores_aux = 0
        # Guarda el valor en el dataframe
        df2.at[df.columns[i], col_names[j]] = diferencia
        # Se imprimen los resultados
    print(df2)
    print(f"Rubros con mayor cantidad de errores para una variación de {col_names[j]}:")
    print(df2[col_names[j]].nsmallest(5))
    print(f"Rubros con menor cantidad de errores para una variación de {col_names[j]}:")
    print(df2[col_names[j]].nlargest(5))
