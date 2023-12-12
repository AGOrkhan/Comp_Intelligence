import random
import time


class Cutting:
    def __init__(self, population, timer, mutation_rate, tournament_size, penalty):
        self.stock_lengths = [10, 13, 15]
        self.stock_costs = [100, 130, 150]
        self.piece_lengths = [3, 4, 5, 6, 7, 8, 9, 10]
        self.quantities = [5, 2, 1, 2, 4, 2, 1, 3]

        self.population = population
        self.timer = timer
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.stocks = list(zip(self.stock_lengths, self.stock_costs))
        self.orders = list(zip(self.piece_lengths, self.quantities))
        self.best_cost = float('inf')

        self.solution = []
        self.wastage = 0
        self.penalty = penalty


def initialize_population():
    population = []
    for _ in range(cut.population):
        individual = generate_individual()
        population.append(individual)
    return population


def generate_individual():
    individual = []
    remaining_orders = dict(cut.orders)

    while any(qty > 0 for qty in remaining_orders.values()):
        selected_stock, stock_cost = random.choice(cut.stocks)
        cutting_plan, remaining_orders = generate_cutting_plan(selected_stock, remaining_orders)
        individual.append((cutting_plan, stock_cost))

    return individual


def generate_cutting_plan(stock_length, remaining_orders):
    cutting_plan = []
    shuffled_orders = list(remaining_orders.items())
    random.shuffle(shuffled_orders)

    for piece_length, quantity in shuffled_orders:
        while quantity > 0 and stock_length >= piece_length:
            cutting_plan.append(piece_length)
            stock_length -= piece_length
            quantity -= 1
        remaining_orders[piece_length] = quantity

    return cutting_plan, remaining_orders


def calculate_fitness(individual):
    total_cost = 0
    penalty = 0
    piece_usage = {length: 0 for length, _ in cut.orders}

    for cutting_plan, stock_cost in individual:
        total_cost += stock_cost
        total_length = sum(cutting_plan)
        stock_length = next((length for length, cost in cut.stocks if cost == stock_cost), None)

        if total_length > stock_length:

            penalty += (total_length - stock_length) * cut.penalty

        for piece_length in cutting_plan:
            piece_usage[piece_length] += 1

    for piece_length, quantities in cut.orders:
        deviation = abs(piece_usage[piece_length] - quantities)
        penalty += deviation * cut.penalty

    total_cost += penalty
    cut.solution, cut.best_cost = (individual, total_cost) if cut.best_cost > total_cost else (cut.solution,
                                                                                               cut.best_cost)

    # print("Cost: ", total_cost)
    return individual, total_cost


def tournament_selection(population):
    parents = []
    while len(parents) < cut.tournament_size:
        parents.append(min(random.sample(population, cut.tournament_size), key=calculate_fitness))
    return parents


def crossover(parents):
    offspring = []
    for parent1, parent2 in zip(parents[::2], parents[1::2]):

        chunk_size = random.randint(1, min(len(parent1), len(parent2)) // 2)
        start_point = random.randint(0, len(parent1) - chunk_size)

        offspring.append(parent1[:start_point] + parent2[start_point:start_point + chunk_size] + parent1[start_point +
                                                                                                         chunk_size:])
        offspring.append(parent2[:start_point] + parent1[start_point:start_point + chunk_size] + parent2[start_point +
                                                                                                         chunk_size:])
        mutator = random.choice(offspring)
        if random.random() < cut.mutation_rate:
            offspring[offspring.index(mutator)] = mutate(mutator)

    return offspring


def mutate(individual):
    cutting_plan_index = random.randint(0, len(individual) - 1)
    cutting_plan, stock_cost = individual[cutting_plan_index]

    stock_length = next((length for length, cost in cut.stocks if cost == stock_cost), None)

    if len(cutting_plan) > 1:
        i, j = random.sample(range(len(cutting_plan)), 2)
        cutting_plan[i], cutting_plan[j] = cutting_plan[j], cutting_plan[i]

    if sum(cutting_plan) > stock_length:
        cutting_plan.pop()

    individual[cutting_plan_index] = (cutting_plan, stock_cost)

    return individual


def evolution():
    population = initialize_population()
    start_time = time.time()
    while time.time() - start_time < cut.timer:
        population = crossover(tournament_selection(population))


cut = Cutting(1000, 10, 1, 500, 10)
evolution()
# print("Best solution: ", cut.solution, "Wastage: ", cut.wastage, "Cost: ", cut.best_cost)
# print("Best solution: ", cut.solution, "Cost: ", cut.best_cost)
print("Best solution: ", cut.solution, "Cost: ", cut.best_cost)
