import numpy as np
import random
import time
import pandas as pd

fread = pd.read_csv('test.csv')

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
        self.size = 0
        self.mutation = 0
        self.greedy_perc = 0


rt = Routing()


def ran_route(population):
    route_set = []
    for i in range(population):
        route = list(range(len(rt.matrix)))
        random.shuffle(route)
        route_set.append(route)
    return route_set


def greedy_ran_route(population):
    route_set = []
    greedy_count = int(population * rt.greedy_perc)

    for i in range(greedy_count):
        start_city = random.choice(range(len(rt.matrix)))
        route = [start_city]
        while len(route) < len(rt.matrix):
            last_city = route[-1]
            next_city = min(range(len(rt.matrix)),
                            key=lambda j: rt.matrix[last_city][j] if j not in route else float('inf'))
            route.append(next_city)
        route_set.append(route)

    for i in range(population - greedy_count):
        route = list(range(len(rt.matrix)))
        random.shuffle(route)
        route_set.append(route)

    return route_set


def route_calc(route):
    cost = 0
    for j in range(len(route) - 1):
        cost += rt.matrix[route[j], route[j + 1]]
    cost += rt.matrix[route[-1], route[0]]
    rt.score, rt.best_route = (cost, route) if rt.score > cost else (rt.score, rt.best_route)
    return route, cost


def mutation(route):
    swap = route.index(random.sample(route, 1)[0])
    swap = - 1 if swap == len(route)-1 else swap
    route[swap], route[swap + 1] = route[swap + 1], route[swap]
    return route


def crossover(candidates):
    routes = []
    pairs = list(zip(candidates, candidates[1:][::2]))
    for pair in range(len(pairs)):
        parent1 = pairs[pair][0]
        parent2 = pairs[pair][1]

        length = len(parent1)
        child = [None] * length

        start, end = sorted(random.sample(range(length), 2))
        child[start:end + 1] = parent1[start:end + 1]

        child_pos = end + 1
        parent2_pos = end + 1
        while None in child:
            if parent2_pos >= length:
                parent2_pos = 0
            if child_pos >= length:
                child_pos = 0
            if parent2[parent2_pos] not in child:
                child[child_pos] = parent2[parent2_pos]
                child_pos += 1
            parent2_pos += 1

        if random.random() < rt.mutation:
            child = mutation(child)
        routes.append(child)
    return routes


def tournament(routes):
    candidates = []

    for i in range(len(routes) * 2):
        values = list(map(route_calc, random.sample(routes, rt.size)))
        route = list([route[0] for route in values])
        costs = list([value[1] for value in values])
        candidates.append(route[costs.index(min(costs))])
    return candidates


def initialization(population, counter, size, mutation_rate, greedy):
    routes = greedy_ran_route(population)
    rt.size = size
    rt.mutation = mutation_rate
    rt.greedy_perc = greedy
    start = time.time()

    while time.time() - start <= counter:
        rt.searches += 1
        # candidates = tournament(routes)
        # routes = crossover(candidates)

        routes = crossover(tournament(routes))


# Parameters( Population > 1, Time Ran, Tournament size, Mutation rate, Greedy Percentage)
initialization(100, 3, 5, 0.08, 0.2)
print("The final route is:", rt.best_route, "The cost is: ", rt.score, "Search count: ", rt.searches)


