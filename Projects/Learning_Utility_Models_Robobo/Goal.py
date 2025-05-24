
class Need:
    def __init__(self, tag: str, peso: float, activacion: float, satisfaccion: float):
        self.tag = tag
        self.peso = peso
        self.activacion = activacion
        self.satisfaccion = satisfaccion


class Drive:
    def __init__(self, evaluacion: float, activacion: float):
        self.evaluacion = evaluacion
        self.activacion = activacion


class Goal:
    def __init__(self, need: Need, drive: Drive, reward):
        self.need = need
        self.drive = drive
        self.reward = reward
        if self.reward == 1:
            self.activacion = 1
        else:
            self.activacion = 0

def GoalSelector(goals):
    # Debe hacer una función que seleccione el goal a ejecutar
    goalActual = max(goals, key=lambda g: g.need.peso)
    return goalActual.need.tag
        





