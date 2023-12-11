import numpy as np
import random
import time
import copy
import pandas as pd

fread = pd.read_csv('ulysses16.csv')

cities = len(fread)
matrix = np.zeros((cities, cities))


def distance_calc(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


for i in range(cities):
    for j in range(cities):
        if i != j:
            city_i = (fread.iloc[i]['x'], fread.iloc[i]['y'])
            city_j = (fread.iloc[j]['x'], fread.iloc[j]['y'])

            matrix[i][j] = distance_calc(city_i, city_j)


class Routing:
    def __init__(self):
        self.matrix = matrix
        # self.route_dict = {}
        self.score, self.best_route = float('inf'), []
        self.searches = 0


rt = Routing()


def route_calc(route):
    cost = 0
    for i in range(len(route) - 1):
        cost += rt.matrix[route[i], route[i + 1]]
    cost += rt.matrix[route[-1], route[0]]
    return cost

    # rt.route_dict[tuple(route)] = cost


def ran_route(count):
    start = time.time()
    l_cost = float('inf')
    while time.time() - start <= count:
        rt.searches += 1
        route = list(range(len(rt.matrix)))
        random.shuffle(route)
        cost = route_calc(route)
        rt.score, rt.best_route = (cost, route) if rt.score > cost else (rt.score, rt.best_route)
    print("Random: ",rt.best_route, rt.score, rt.searches)


    # Restart for swap
    start = time.time()
    rt.score, rt.best_route = float('inf'), []
    rt.searches = 0

    while time.time() - start <= count:
        route = list(range(len(rt.matrix)))
        random.shuffle(route)
        swap_route(route)
    print("Swap: ", rt.best_route, rt.score, rt.searches)

    # Restart for swap
    start = time.time()
    rt.score, rt.best_route = float('inf'), []
    rt.searches = 0
    while time.time() - start <= count:
        route = list(range(len(rt.matrix)))
        random.shuffle(route)
        opt_route(route)
    print("Opt: ", rt.best_route, rt.score, rt.searches)


def swap_route(route):
    rt.searches += 1
    l_cost = float('inf')
    for i in range(1, len(route)):
        for j in range(i + 1, len(route)):
            copy_route = copy.deepcopy(route)
            copy_route[i], copy_route[j] = copy_route[j], copy_route[i]
            # if tuple(route) not in rt.route_dict:
            cost = route_calc(copy_route)
            l_cost, route = (cost, copy_route) if l_cost > cost else (l_cost, route)
            rt.score, rt.best_route = (cost, route) if rt.score > cost else (rt.score, rt.best_route)


def opt_route(route):
    rt.searches += 1

    l_score = float('inf')
    l_route = route

    made_improvement = True
    while made_improvement:
        made_improvement = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_cost = route_calc(new_route)

                if new_cost < l_score:
                    l_route = new_route
                    l_score = new_cost
                    made_improvement = True

        route = l_route
    rt.score, rt.best_route = (l_score, route) if rt.score > l_score else (rt.score, rt.best_route)


# Put time in parameter
ran_route(3)

