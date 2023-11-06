import numpy as np
import random

class Routing:
    def __init__(self):
        self.matrix = np.zeros((4, 4))


rt = Routing()

rt.matrix[0, 1] = rt.matrix[1, 0] = 20
rt.matrix[0, 2] = rt.matrix[2, 0] = 42
rt.matrix[0, 3] = rt.matrix[3, 0] = 35
rt.matrix[1, 2] = rt.matrix[2, 1] = 30
rt.matrix[1, 3] = rt.matrix[3, 1] = 34
rt.matrix[2, 3] = rt.matrix[3, 2] = 12


def calc_route(matrix, route):
    cost = 0
    for i in range(len(route) - 1):
        cost += matrix[route[i], route[i+1]]
    cost += matrix[route[-1], route[0]]

    print(route, cost)


def ran_route():
    route = list(range(len(rt.matrix)))
    random.shuffle(route)
    calc_route(rt.matrix, route)


ran_route()




