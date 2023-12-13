import random
import time
import matplotlib.pyplot as matlib
import concurrent.futures


class Cutting:
    def __init__(self, population, timer, mutation_rate, tournament_size, mutation, niche_size, penalty_multiplier):
        self.stock_lengths = [4300, 4250, 4150, 3950, 3800, 3700, 3550, 3500]
        self.stock_costs = [86, 85, 83, 79, 68, 66, 64, 63]
        self.piece_lengths = [2350, 2250, 2200, 2100, 2050, 2000, 1950, 1900, 1850, 1700, 1650, 1350, 1300, 1250, 1200, 1150, 1100, 1050]
        self.quantities = [2, 4, 4, 15, 6, 11, 6, 15, 13, 5, 2, 9, 3, 6, 10, 4, 8, 3]

        self.population = population * niche_size
        self.timer = timer
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.mutation_scale = mutation
        self.niche_size = niche_size
        self.penalty_multiplier = penalty_multiplier
        self.generations = 0

        self.stocks = list(zip(self.stock_lengths, self.stock_costs))
        self.orders = list(zip(self.piece_lengths, self.quantities))

        self.best_cost = float('inf')
        self.solution = []


def initialize_population():
    population = []
    for _ in range(cut.population):
        individual = generate_individual()
        population.append(individual)
    return population


def generate_individual():
    individual = {
        'stock_details': [],
        'total_cost': float('inf')
    }
    remaining_orders = dict(cut.orders)

    while any(qty > 0 for qty in remaining_orders.values()):
        selected_stock, stock_cost = random.choice(cut.stocks)
        cutting_plan, remaining_orders = generate_cutting_plan(selected_stock, remaining_orders)
        individual['stock_details'].append((selected_stock, cutting_plan, stock_cost))
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

    for stock_length, cutting_plan, stock_cost in individual['stock_details']:
        total_cost += stock_cost

        if sum(cutting_plan) > stock_length:
            penalty += (sum(cutting_plan) - stock_length) * cut.penalty_multiplier

        for piece_length in cutting_plan:
            piece_usage[piece_length] += 1

    for piece_length, quantities in cut.orders:
        deviation = abs(piece_usage[piece_length] - quantities)
        penalty += deviation * cut.penalty_multiplier

    total_cost += penalty
    individual['total_cost'] = total_cost

    cut.solution, cut.best_cost = (individual, total_cost) if cut.best_cost > total_cost else (
        cut.solution, cut.best_cost)


def tournament_selection(population):
    parents = []
    while len(parents) < cut.tournament_size:
        parents.append(min(random.sample(population, cut.tournament_size),
                           key=lambda individual: individual['total_cost']))
    return parents


def crossover(parents):
    offspring = []
    for parent1dict, parent2dict in zip(parents[::2], parents[1::2]):
        parent1 = parent1dict['stock_details']
        parent2 = parent2dict['stock_details']

        chunk_size = random.randint(1, min(len(parent1), len(parent2)) // 2)
        start_point = random.randint(0, len(parent1) - chunk_size)

        offspring_chunk1 = (parent1[:start_point] + parent2[start_point:start_point + chunk_size]
                            + parent1[start_point + chunk_size:])
        offspring_chunk2 = (parent2[:start_point] + parent1[start_point:start_point + chunk_size]
                            + parent2[start_point + chunk_size:])

        offspring1 = {'stock_details': offspring_chunk1, 'total_cost': float('inf')}
        offspring2 = {'stock_details': offspring_chunk2, 'total_cost': float('inf')}

        offspring.append(offspring1)
        offspring.append(offspring2)

        for individual in offspring:
            if random.random() < cut.mutation_rate:
                mutate(individual)

    return offspring


def mutate(individual):
    cutting_plan_index = random.randint(0, len(individual) - 1)
    stock_length, cutting_plan, stock_cost = individual['stock_details'][cutting_plan_index]

    for _ in range(cut.mutation_scale):
        if len(cutting_plan) > 1:
            i, j = random.sample(range(len(cutting_plan)), 2)
            cutting_plan[i], cutting_plan[j] = cutting_plan[j], cutting_plan[i]

    if sum(cutting_plan) > stock_length:
        cutting_plan.pop()

    individual['stock_details'][cutting_plan_index] = (stock_length, cutting_plan, stock_cost)
    return individual


def dividing(population):
    population.sort(key=lambda individual: individual['total_cost'])

    # Divide into niches
    niches = []
    niche_size = len(population) // cut.niche_size
    for i in range(cut.niche_size):
        niche = population[i * niche_size:(i + 1) * niche_size]
        niches.append(niche)

    return niches

    # niche_size = len(population) // cut.niche_size
    #
    # niches = []
    # for _ in range(cut.niche_size):
    #     niche = random.sample(population, niche_size)
    #     niches.append(niche)
    # return niches


def evolution():
    # Reset if necessary
    cut.best_cost = float('inf')
    cut.solution = []
    cut.generations = 0

    # Evolutionary algorithm
    population = initialize_population()
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=cut.niche_size) as executor:
        while time.time() - start_time < cut.timer:
            processes = []
            new_pop = []
            cut.generations += 1
            [calculate_fitness(individual) for individual in population]
            for niche in dividing(population):
                process = executor.submit(crossover, tournament_selection(niche))
                processes.append(process)

            for future in concurrent.futures.as_completed(processes):
                child = process.result()
                new_pop.extend(child)

            population = new_pop


cut = Cutting(200, 10, 0.1, 50, 3, 3, 100)
evolution()
