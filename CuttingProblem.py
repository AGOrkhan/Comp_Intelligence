import random
import time


class Cutting:
    def __init__(self, population, timer, mutation_rate, crossover_rate, tournament_size):
        self.stock_lengths = [10, 13, 15]
        self.stock_costs = [100, 130, 150]
        self.piece_lengths = [3, 4, 5, 6, 7, 8, 9, 10]
        self.quantities = [5, 2, 1, 2, 4, 2, 1, 3]
        self.population = population
        self.timer = timer
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size * 2
        self.stocks = list(zip(self.stock_lengths, self.stock_costs))
        self.orders = list(zip(self.piece_lengths, self.quantities))
        self.best_cost = float('inf')
        self.solution = []
        self.wastage = 0


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

    # Shuffle the remaining orders
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
    total_wastage = 0

    for cutting_plan, stock_cost in individual:
        total_cost += stock_cost
        """used_length = sum(cutting_plan)
        stock_length = next((i[0] for i in cut.stocks if i[1] == stock_cost))
        total_wastage += stock_length - used_length"""
    # print("Solution: ", individual, "Wastage: ", total_wastage, "Cost: ", total_cost)
    return individual, total_cost


def tournament_selection(population):
    parents = []
    for _ in range(cut.tournament_size):
        parents.append(min(random.sample(population, cut.tournament_size), key=calculate_fitness))
    return parents



def single_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    return [offspring1, offspring2]


def mutate(individual):
    if random.random() < cut.mutation_rate:
        mutation_point = random.randint(0, len(individual) - 1)

        # Correctly identify the stock length for the mutation point
        stock_length = individual[mutation_point][0]
        stock_length = stock_length[0]

        new_cutting_plan, _ = generate_cutting_plan(stock_length, dict(cut.orders))
        individual[mutation_point] = (new_cutting_plan, individual[mutation_point][1])

    return individual



def evolution():
    population = initialize_population()
    start_time = time.time()

    while time.time() - start_time < cut.timer:
        new_population = []

        while len(new_population) < cut.population:
            # Selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            # Crossover
            offspring1, offspring2 = single_point_crossover(parent1, parent2)

            # Mutation
            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)

            # Add offspring to the new population
            new_population.extend([offspring1, offspring2])

        # Ensure the population size is constant
        population = new_population[:cut.population]

        # Update the best solution
        for individual in population:
            cost, wastage = calculate_fitness(individual)
            if cost < cut.best_cost:
                cut.best_cost = cost
                cut.solution = individual
                cut.wastage = wastage


cut = Cutting(1000, 3, 0.01, 0.9, 2)
evolution()
# print("Best solution: ", cut.solution, "Wastage: ", cut.wastage, "Cost: ", cut.best_cost)
# print("Best solution: ", cut.solution, "Cost: ", cut.best_cost)
