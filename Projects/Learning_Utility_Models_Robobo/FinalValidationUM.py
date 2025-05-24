from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
from collections import deque
from tensorflow.keras.models import load_model

from scipy.spatial import distance

import math
import time
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from Goal import *




# This creates an instance of the Robobo class with the localhost IP address
IP = "localhost"
rob = Robobo(IP)
rob.connect()
sim = RoboboSim(IP)
sim.connect()


def predictTurns(action, yaw):
    yaw_ranges = [
        (-22.5, 22.5), (67.5, 112.5), (157.5, 180), (-180, -157.5), (-112.5, -67.5),  
        (22.5, 67.5), (112.5, 157.5), (-157.5, -112.5), (-67.5, -22.5)
    ]
    yaw_mappings = {
        90:    [90.0, 180.0, -90.0, -90.0, 0.0, 135.0, -135.0, -45.0, 45.0],
        -90:   [-90.0, 0.0, 90.0, 90.0, 180.0, -45.0, 45.0, 135.0, -135.0],
        45:    [45.0, 135.0, -135.0, -135.0, -45.0, 90.0, 180.0, -90.0, 0.0],
        -45:   [-45.0, 45.0, 135.0, 135.0, -135.0, 0.0, 90.0, 180.0, -90.0]
    }

    for i, (low, high) in enumerate(yaw_ranges):
        if low <= yaw <= high:
            return yaw_mappings[action][i]
    return None

def predictForward(x, z, yaw):
    # Rango de posiciones
    in_bounds = -700 < x < 700 and -700 < z < 700
    
    # Tabla de movimientos en función del yaw
    directions = [
        ((-22.5, 22.5), (-130, 0)), 
        ((67.5, 112.5), (0, -130)), 
        ((157.5, 180), (130, 0)), 
        ((-180, -157.5), (130, 0)),  
        ((-112.5, -67.5), (0, 130)), 
        ((22.5, 67.5), (-85, -105)),  
        ((112.5, 167.5), (85, -105)),  
        ((-167.5, -112.5), (85, 105)),  
        ((-67.5, -22.5), (-85, 105)),   
    ]
    for (yaw_min, yaw_max), (dx, dz) in directions:
        if yaw_min <= yaw <= yaw_max and (
            abs(dx) == 130 or in_bounds  
        ):
            return x + dx, z + dz
    return x, z 

def distanceC(xr, zr, xg, zg, xb, zb, x2, z2):
    red = math.sqrt((xr - x2)**2 + (zr - z2)**2)
    green = math.sqrt((xg - x2)**2 + (zg - z2)**2)
    blue = math.sqrt((xb - x2)**2 + (zb - z2)**2)
    return red, green, blue

def ErrorWM (red, green, xn, zn, xr, zr, xg, zg):
    realred = math.sqrt((xr - xn)**2 + (zr - zn)**2)
    realgreen = math.sqrt((xg - xn)**2 + (zg - zn)**2)
    redError = (red - realred)
    greenError = (green - realgreen)
    return redError, greenError

def Avanzar(distanciar, distanciag, distanciab):
    rob.moveWheels(15, 15)  # Avanzar
    Stime = time.time()
    while rob.readIRSensor(IR.FrontC) < 50 and (time.time() - Stime) < 1.5:
        rob.wait(0.1)   
    rob.stopMotors()
    if rob.readIRSensor(IR.FrontC) > 60 and distanciab < 100 or distanciag < 100 or distanciar < 100: 
        rob.stopMotors()
        return 
    rob.wait(0.1)
    return 

def getRobotPos():
    loc = sim.getRobotLocation(0)
    pos = loc['position']
    x = pos["x"]
    z = pos["z"]
    return x, z

def predictMoves(actions, xr, zr, xg, zg, xb, zb):
    possible_moves = pd.DataFrame(columns=["Red Distance", "Green Distance", "Blue Distance", "Action"])
    yaw = rob.readOrientationSensor().yaw
    n = 0
    for action in actions:
        x, z = getRobotPos()
        if action == 0:
            x2, z2 = predictForward(x, z, yaw)
            new_red, new_green, new_blue = distanceC(xr, zr, xg, zg, xb, zb, x2, z2)
        else:
            yaw2 = predictTurns(action, yaw)
            x2, z2 = predictForward(x, z, yaw2)
            new_red, new_green, new_blue = distanceC(xr, zr, xg, zg, xb, zb, x2, z2)
        possible_moves.loc[n] = ([new_red, new_green, new_blue, action])
        n += 1
    # Nomalizar los datos
    min_val = 0
    max_val = np.sqrt(8000000)

    # Normalizar solo las columnas de distancia
    possible_moves["Red Distance"] = (possible_moves["Red Distance"] - min_val) / (max_val - min_val)
    possible_moves["Green Distance"] = (possible_moves["Green Distance"] - min_val) / (max_val - min_val)
    possible_moves["Blue Distance"] = (possible_moves["Blue Distance"] - min_val) / (max_val - min_val)
    return possible_moves
    
def graficarResultado(todos, reales):
    # Definir los valores mínimo y máximo
    min_val = 0
    max_val = np.sqrt(8000000)

    # Normalizar solo las columnas de distancia
    reales["Red Distance"] = (reales["Red Distance"] - min_val) / (max_val - min_val)
    reales["Green Distance"] = (reales["Green Distance"] - min_val) / (max_val - min_val)
    #reales = np.array(reales)
    plt.figure(figsize=(8, 6))
    
    plt.scatter(reales["Red Distance"], reales["Green Distance"], color='blue', label='Set Azul')
    plt.scatter(todos["Red Distance"], todos["Green Distance"], color='red')
    #plt.text(todos["Red Distance"], todos["Green Distance"], str(todos['Score']), fontsize=12, ha='right', color='black')
    plt.xlim(0, 0.8)  
    plt.ylim(0, 0.8)  

    plt.show()
    return

def random_from_intervals():
    if random.random() < 0.5:
        return random.randint(-700, -100)  # Intervalo negativo
    else:
        return random.randint(100, 700)    # Intervalo positivo
    
def ChangeObject(x, z, i):

    locY = sim.getObjectLocation('CUSTOMCYLINDER')
    # Move the selected object +50mm in the X axis
    posY = locY['position']
    posY["x"] = 1200
    posY["z"] = 1
    sim.setObjectLocation("CUSTOMCYLINDER", locY['position'])

    objects = ['REDCYLINDER', 'GREENCYLINDER','BLUECYLINDER']
    object = objects[i]
    loc2 = sim.getObjectLocation(object)
    # Move the selected object +50mm in the X axis
    pos2 = loc2['position']
    pos2["x"] = x
    pos2["z"] = z
    sim.setObjectLocation(object, loc2['position'])
    sim.wait(0.5)
    return

def getObjectLocation():
    locr = sim.getObjectLocation('REDCYLINDER')
    # Move the selected object +50mm in the X axis
    posr = locr['position']
    xr = posr["x"]
    zr = posr["z"]

    locg = sim.getObjectLocation('GREENCYLINDER')
    # Move the selected object +50mm in the X axis
    posg = locg['position']
    xg = posg["x"]
    zg = posg["z"]

    locb = sim.getObjectLocation('BLUECYLINDER')
    # Move the selected object +50mm in the X axis
    posb = locb['position']
    xb = posb["x"]
    zb = posb["z"]
    return xr, zr, xg, zg, xb, zb

def evaluarPred(candidates, network):
    df1 = candidates.drop(columns=["Action"])     
    valuation = network.predict(df1)
    candidates['Score'] = valuation
    # Ordenor los estados evaluados
    df_sorted = candidates.sort_values(by="Score", ascending=False)  # Ordena por la última columna

    return df_sorted

def EncuentraObjetivo(objetivo, dr, dg, db):
    encontrado =True if (objetivo == "R" and dr < 150) or (objetivo == "G" and dg < 150) or (objetivo == "B" and db < 150) else False
    return encontrado

def ReachGoal(model, tag):
    actions = [0, 90,-90, 45, -45]
    encontrado = False
    xr, zr, xg, zg, xb, zb = getObjectLocation()
    x, z = getRobotPos()
    realDr, realDg, realDb = distanceC(xr, zr, xg, zg, xb, zb, x, z)
    while encontrado == False:
        possible_moves = predictMoves(actions, xr, zr, xg, zg, xb, zb)
        moves = evaluarPred(possible_moves, model)
        print(moves)
        new_red = moves.iloc[0, 0]
        new_green = moves.iloc[0, 1]
        new_blue = moves.iloc[0, 2]
        actionR = moves.iloc[0, 3]
        if actionR == 90:
            rob.moveWheelsByTime(5, -5, 3.65, wait=True)
            rob.wait(0.1)
        elif actionR == -90:
            rob.moveWheelsByTime(-5, 5, 3.65, wait=True)
            rob.wait(0.1)
        elif actionR == 45:
            rob.moveWheelsByTime(5, -5, 2, wait=True)
            rob.wait(0.1)
        elif actionR == -45:
            rob.moveWheelsByTime(-5, 5, 2, wait=True)
            rob.wait(0.1)   
        Avanzar(new_red, new_green, new_blue)
        rob.wait(0.1)
        xn, zn = getRobotPos()
        realDr, realDg, realDb= distanceC(xr, zr, xg, zg, xb, zb, xn, zn)
        rob.wait(0.1)
        encontrado = EncuentraObjetivo(tag, realDr, realDg, realDb)
    return encontrado

def distanciaE(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)



networkR = load_model("C:\\Users\\diana\\ANNRed.keras")
networkG = load_model("C:\\Users\\diana\\ANNG.keras")
networkB = load_model("C:\\Users\\diana\\ANNB.keras")
file_path = "C:\\Users\\diana\\Utility.csv"

picked_points = []  # Lista para almacenar los puntos (x, z) seleccionados
distancia_minima = 500  # Distancia mínima requerida entre los puntos

for i in range(3):
    while True:
        x = random_from_intervals()
        z = random_from_intervals()
        nuevo_punto = (x, z)
        if all(distanciaE(nuevo_punto, p) > distancia_minima for p in picked_points):
            picked_points.append(nuevo_punto)  
            ChangeObject(x, z, i)  
            break


###############################################################
# DEFINICIÓN DE NEEDS, DRIVES, GOAL
# NEED 1
needR = Need(tag="R", peso=1, activacion=1, satisfaccion=0)
driveR = Drive(evaluacion=0, activacion=1)
Rgoal = Goal(needR, driveR, 0)

# NEED 2
needG = Need(tag="G", peso=0.8, activacion=1, satisfaccion=0)
driveG = Drive(evaluacion=0, activacion=1)
Ggoal = Goal(needG, driveG, 0)

# NEED 3
needB = Need(tag="B", peso=0.7, activacion=1, satisfaccion=0)
driveB = Drive(evaluacion=0, activacion=1)
Bgoal = Goal(needB, driveB, 0)


goals = [Rgoal, Ggoal, Bgoal]

##############################################################
sim.wait(2)
task = True
while task:
    objetivo = GoalSelector(goals)
    if objetivo == "R":
        goalLogrado = ReachGoal(networkR, objetivo)
        Rgoal.need.peso = 0
    elif objetivo == "G":
        goalLogrado = ReachGoal(networkG, objetivo)
        Ggoal.need.peso = 0
    elif objetivo == "B":
        goalLogrado = ReachGoal(networkB, objetivo)
        Bgoal.need.peso = 0
    
    task = not all(g.need.peso == 0 for g in goals)
sim.disconnect()
rob.disconnect()


