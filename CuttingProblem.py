import random
import time


class Cutting:
    def __init__(self, population, timer, mutation_rate, tournament_size):
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
    # total_wastage = 0

    for cutting_plan, stock_cost in individual:
        total_cost += stock_cost

    cut.solution, cut.best_cost = (individual, total_cost) if cut.best_cost > total_cost else (cut.solution,
                                                                                                   cut.best_cost)

    # cut.wastage = wastage
    """used_length = sum(cutting_plan)
    stock_length = next((i[0] for i in cut.stocks if i[1] == stock_cost))
    total_wastage += stock_length - used_length"""
    print("Solution: ", individual, "Cost: ", total_cost)
    return individual, total_cost


def tournament_selection(population):
    parents = []
    for _ in range(cut.tournament_size):
        parents.append(min(random.sample(population, cut.tournament_size), key=calculate_fitness))
    return parents


def crossover(parents):
    offspring = []
    for parent1, parent2 in zip(parents[::2], parents[1::2]):

        midpoint = len(parent1) // 2
        child1 = parent1[:midpoint] + parent2[midpoint:]
        child2 = parent2[:midpoint] + parent1[midpoint:]

        offspring.append(adjust_offspring(child1))
        offspring.append(adjust_offspring(child2))

    return offspring


def adjust_offspring(offspring):
    required_quantities = dict(cut.orders)
    current_quantities = {length: 0 for length, _ in cut.orders}

    for cutting_plan, _ in offspring:
        for length in cutting_plan:
            current_quantities[length] += 1

    for cutting_plan, stock_cost in offspring:
        for i, length in enumerate(cutting_plan):
            if current_quantities[length] > required_quantities[length]:
                for replacement_length, qty in current_quantities.items():
                    if qty < required_quantities[replacement_length] and replacement_length <= length:
                        cutting_plan[i] = replacement_length
                        current_quantities[length] -= 1
                        current_quantities[replacement_length] += 1
                        break
    return offspring


def evolution():
    population = initialize_population()
    start_time = time.time()
    while time.time() - start_time < cut.timer:
        population = crossover(tournament_selection(population))


cut = Cutting(1000, 2, 0.01, 10)
evolution()
# print("Best solution: ", cut.solution, "Wastage: ", cut.wastage, "Cost: ", cut.best_cost)


# Assuming cut.solution is set
def count_lengths_in_solution(solution):
    length_usage = {}

    for cutting_plan, _ in solution:
        for length in cutting_plan:
            if length in length_usage:
                length_usage[length] += 1
            else:
                length_usage[length] = 1

    return length_usage

print("Best solution: ", cut.solution, "Cost: ", cut.best_cost)
