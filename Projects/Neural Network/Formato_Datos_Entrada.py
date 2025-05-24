# Importación de librerías
import pandas as pd
import numpy as np


# Definición de funciones
# Función para normalizar los datos en un rango de 0 a 1
def normalizing(df):
    max_val = df.max(axis=1) # Se obtiene el máximo de cada columna
    min_val = df.min(axis=1) # Se obtiene el mínimo de cada columna
    range = max_val - min_val # Se obtiene la diferencia de los dos
    df = (df - min_val)/(range) # Y se utiliza para normalizarlas
    df = df.astype(float) # Se asegura que los datos sean tipo float
    return df

# Función utilizada para formatear y ordenar los datos en un dataframe
def formatting(df):
    df = df.reset_index(drop=False)
    df.pop("index")
    # Crea las columnas necesarias para la entrada
    for i in range(14, 0, -1):
        df.insert(7, f"Tz{i+1}", " ")
        df.insert(7, f"Ty{i+1}", " ")
        df.insert(7, f"Tx{i+1}", " ")
        df.insert(7, f"Fz{i+1}", " ")
        df.insert(7, f"Fy{i+1}", " ")
        df.insert(7, f"Fx{i+1}", " ")
    n_datasets = len(df.index)/16 # Saca cuantos datos hay en el documento para repetir el acomodo
    for k in range(0, int(n_datasets)):
        for j in range(1, 16):
            df.loc[k*16, df.columns[1+(6*(j-1))]:df.columns[6+(6*(j-1))]] = df.loc[j+k*16, df.columns[1]:df.columns[6]].to_numpy()
            df = df.drop(index=j+16*k)
    df = df.rename(columns={'Tipo': 'Resultado'})
    df = df.replace(to_replace="normal", value=1) # Se remplazan los valores de normal por 0.0001
    df = df.replace(to_replace="collision", value=3) #1
    df = df.replace(to_replace="obstruction", value=2) #0.1
    df = df.replace(to_replace="fr_collision", value=0) #0.01
    df = normalizing(df) # Se normalizan los datos por columnas
    df = df.replace(0.000000, 0.00001) # Elimina los ceros para que no hay problemas con la correción
    return df

###################### FORMATEO DE DATOS Y CREACIÓN DE ARCHIVO DE EXCEL ###########################
# La siguiente sección de código se usó para formatear los datos y cargarlos en un archivo de Excel
# Se debe ejecutar solo una vez para generar el archivo .cvs, posteriormente se debe comentar
df1 = pd.read_csv("lp1_1.data", sep='\t', lineterminator='\n')
df1 = formatting(df1)
df2 = pd.read_csv("lp4.data", sep='\t', lineterminator='\n')
df2 = formatting(df2)
data = pd.concat([df1, df2], axis=0) # Concatena todos los grupos de datos en un sólo DataFrame
data = data.reset_index() # Resetea el index para que quede en orden después de concatenar
data.pop("index") # Elimina una columna innecesaria
data.to_csv("All_data.csv") # Guarda el dataframe en excel
# Se crea un excel con los resultados para determinar cuántos datos hay en cada categoría
#arreglo = data["Resultado"]
#f = pd.DataFrame(arreglo)
#f.to_csv("tipos.csv", index=False)


