from deap import base, creator, tools, algorithms
import random
import numpy
import array
import matplotlib.pyplot as plt
import math


K1 = 110563.25
sin_60 = math.sqrt(3)/2
SIGMA_Y = 205e6 # 205 MPa


def Prueba(individuo):
    # Fuanción utilizada para obtener los factores de seguridad del mejor
    # individuo del algoritmo
    # Se extraen los valores de diámetros desde el individuo.
    d_ab = individuo[0]
    d_ae = individuo[1]
    d_bc = individuo[2]
    d_be = individuo[3]
    d_cd = individuo[4]
    d_ce = individuo[5]
    d_de = individuo[6]
    # Se calcula el esfuerzo en cada una de las vigas.
    e_ab = (4/numpy.pi)*((((K1/20*sin_60)*((2.5*d_ab**2) - (10*d_bc**2) - (12.5*d_be) 
                                           + (5*d_ae**2) - (7.5*d_ce**2) + (5*d_de**2) 
                                           + (7.5*d_cd**2)))- 2886.75)/d_ab**2)
    
    e_ae = (4/numpy.pi)*((((-(K1*0.5)/20*sin_60)*(2.5*d_ab**2 - 10*d_bc**2 - 12.5*d_be 
                                                + 5*d_ae**2 - 7.5*d_ce**2 + 5*d_de**2 + 
                                                7.5*d_cd**2))+1443.38)/d_ae**2)
    
    e_be = (4/numpy.pi)*((((-K1/20*sin_60)*(12.5*d_ab**2 - 2.5*d_be**2 + 5*d_ae**2 -
                                            7.5*d_ce**2 + 5*d_de**2 + 7.5*d_cd**2))+2886.75)/d_be**2)
    
    e_ed = (4/numpy.pi)*((K1*(0.5)/(20*sin_60))*((2.5*d_ab**2)+(10*d_bc**2)+(7.5*d_be**2)+
                                                 (5*d_ae**2)+(12.5*d_ce**2)+(5*d_de**2)+
                                                 (7.5*d_cd**2))+1443.38)/(d_de**2)
    
    e_cd = (4/numpy.pi)*((-K1/(20*sin_60))*((2.5*d_ab**2)+(10*d_bc**2)+(7.5*d_be**2)+
                                            (5*d_ae**2)+(12.5*d_ce**2)+(5*d_de**2)+
                                            (7.5*d_cd**2))-2886.75)/(d_cd**2)
    e_ec = (4/numpy.pi)*((K1/(20*sin_60))*((2.5*d_ab**2)+(7.5*d_be**2)+(5*d_ae**2)+
                                           (2.5*d_ce**2)+(5*d_de**2)-(2.5*d_cd**2))+2886.75)/(d_ce**2)
    
    e_bc = (4/numpy.pi)*((-K1*(0.5)/(20*sin_60))*((5*d_ab**2)+(10*d_bc**2)+(15*d_be**2)+
                                                  (10*d_ae**2)+(15*d_ce**2)+(10*d_de**2)+
                                                  (5*d_cd**2))-2886.75)/(d_bc**2)
    
    # Se calcula el Factor de Seguridad en cada una de las vigas.
    FS_ab = SIGMA_Y/abs(e_ab)
    FS_ae = SIGMA_Y/abs(e_ae)
    FS_be = SIGMA_Y/abs(e_be)
    FS_ed = SIGMA_Y/abs(e_ed)
    FS_cd = SIGMA_Y/abs(e_cd)
    FS_ec = SIGMA_Y/abs(e_ec)
    FS_bc = SIGMA_Y/abs(e_bc)
    FS_array = [FS_ab, FS_ae, FS_be, FS_ed, FS_cd, FS_ec, FS_bc]
    print(FS_array)

def funcionEval(individuo):
    # Función que calcula la calidad de un individuo y los factores de
    # seguridad de los elementos con base en las ecuaciones estáticas obtenidas
    # Se extraen los valores de diámetros desde el individuo.
    d_ab = individuo[0]
    d_ae = individuo[1]
    d_bc = individuo[2]
    d_be = individuo[3]
    d_cd = individuo[4]
    d_ce = individuo[5]
    d_de = individuo[6]
    # Se calcula el esfuerzo en cada una de las vigas.
    e_ab = (4/numpy.pi)*((((K1/20*sin_60)*((2.5*d_ab**2) - (10*d_bc**2) - (12.5*d_be) +
                                           (5*d_ae**2) - (7.5*d_ce**2) + (5*d_de**2) +
                                           (7.5*d_cd**2)))- 2886.75)/d_ab**2)
    
    e_ae = (4/numpy.pi)*((((-(K1*0.5)/20*sin_60)*(2.5*d_ab**2 - 10*d_bc**2 - 12.5*d_be
                                                  + 5*d_ae**2 - 7.5*d_ce**2 + 5*d_de**2
                                                  + 7.5*d_cd**2))+1443.38)/d_ae**2)
    
    e_be = (4/numpy.pi)*((((-K1/20*sin_60)*(12.5*d_ab**2 - 2.5*d_be**2 + 5*d_ae**2 -
                                            7.5*d_ce**2 + 5*d_de**2 + 7.5*d_cd**2))+2886.75)/d_be**2)
    
    e_ed = (4/numpy.pi)*((K1*(0.5)/(20*sin_60))*((2.5*d_ab**2)+(10*d_bc**2)+(7.5*d_be**2)
                                                 +(5*d_ae**2)+(12.5*d_ce**2)+(5*d_de**2)
                                                 +(7.5*d_cd**2))+1443.38)/(d_de**2)
    
    e_cd = (4/numpy.pi)*((-K1/(20*sin_60))*((2.5*d_ab**2)+(10*d_bc**2)+(7.5*d_be**2)+(5*d_ae**2)+
                                            (12.5*d_ce**2)+(5*d_de**2)+(7.5*d_cd**2))-2886.75)/(d_cd**2)
    
    e_ec = (4/numpy.pi)*((K1/(20*sin_60))*((2.5*d_ab**2)+(7.5*d_be**2)+(5*d_ae**2)+
                                           (2.5*d_ce**2)+(5*d_de**2)-(2.5*d_cd**2))+2886.75)/(d_ce**2)
    
    e_bc = (4/numpy.pi)*((-K1*(0.5)/(20*sin_60))*((5*d_ab**2)+(10*d_bc**2)+(15*d_be**2)+(10*d_ae**2)+
                                                  (15*d_ce**2)+(10*d_de**2)+(5*d_cd**2))-2886.75)/(d_bc**2)
    
    # Se calcula el Factor de Seguridad en cada una de las vigas.
    FS_ab = SIGMA_Y/abs(e_ab)
    FS_ae = SIGMA_Y/abs(e_ae)
    FS_be = SIGMA_Y/abs(e_be)
    FS_ed = SIGMA_Y/abs(e_ed)
    FS_cd = SIGMA_Y/abs(e_cd)
    FS_ec = SIGMA_Y/abs(e_ec)
    FS_bc = SIGMA_Y/abs(e_bc)
    FS_array = [FS_ab, FS_ae, FS_be, FS_ed, FS_cd, FS_ec, FS_bc]
    
    if not all(i >= 1.5 for i in FS_array) or not all(j >= 0 for j in individuo):
        error = 1
    else:
        error = sum(individuo)
    return [error, ]

if __name__ == "__main__":
    lim_inferior = 0
    lim_superior = 0.1
    poblacion = 5000
    n_generaciones = 200
    probabilidad_mutacion_ind = 0.2 # Es partícular de la mutación Gaussiana
    probabilidad_mutacion_gen = 0.05 # Este es el hiperparámetro
    # Definición del toolbox
    toolbox = base.Toolbox()
    # El fitness que manejará el toolbox será una función con el peso 1
    # (maximización con peso unitario para cada atributo)
    creator.create('Fitness_funct', base.Fitness, weights=(-1.0,))
    # -El individuo que manejará el toolbox será un array de floats
    creator.create('Individuo', array.array,
    fitness=creator.Fitness_funct, typecode='f')
    # Registro de la función de evaluación, usando lo definido previamente en el código
    toolbox.register('evaluate', funcionEval)
    # Registro del método de cómo generar un atributo para este caso: floats al azar
    toolbox.register('atributo', random.uniform, a=lim_inferior,
    b=lim_superior)
    # Se genera un atributo de float al azar 3 veces, y se guarda en un Individuo
    toolbox.register('individuo_gen', tools.initRepeat, creator.Individuo,
    toolbox.atributo, n=7)
    # Luego, se registra en toolbox una operación para crear la población
    toolbox.register('Poblacion', tools.initRepeat, list,
    toolbox.individuo_gen, n=poblacion)
    # Para ello, llama unas 30 veces a la función 'individuo_gen', de manera
    # que queda generada una población de 'Individuo's.
    # Se utiliza la función registrada para generar una población
    popu = toolbox.Poblacion()
    # Método de cruce de dos puntos
    toolbox.register('mate', tools.cxUniform, indpb=0.5)
    # Para la mutación, se utiliza el método de mutación Gaussiana
    toolbox.register('mutate', tools.mutShuffleIndexes,
    indpb=probabilidad_mutacion_ind)
    # Para la selección, se utiliza el método de torneo y el de mejor calidad
    toolbox.register('select', tools.selTournament, tournsize=2)
    #toolbox.register('select', tools.selBest)
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
    popu, logbook = algorithms.eaSimple(popu, toolbox, cxpb=0.2,
                                        mutpb=probabilidad_mutacion_gen,
                                        ngen=n_generaciones, stats=stats,
                                        halloffame=hof, verbose=True)
    print(popu)
    print('---------------------------')
    print(logbook)
    # Gráficas de los resultados
    # X axis parameter:
    xaxis = logbook.select("gen")
    # Y axis parameter:
    quality = logbook.select("min")
    std = logbook.select("std")
    plt.plot(xaxis, std, label='Desviación Estándar')
    plt.plot(xaxis, quality, label='Calidad (Fitness)')
    plt.xlabel('Generación')
    plt.ylabel('Valor')
    plt.legend()
    plt.show()
    # Se imprime el último valor de calidad y desviación estándar
    print('{0:.5f}'.format(funcionEval(hof[0])[0]))
    print('{0:.5f}'.format(std[-1]))

