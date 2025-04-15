# Non-dominated Sorting Genetic Algorith II (NSGA-II)
import random


class NSGAII:
    def __init__(
        self,
        op,
        better=min,
        callback=None,
        n_generations=500,
        population_size=100,
        mutation_rate=0.15,
        crossover_rate=0.95,
        verbose=False,
        return_history=False,
    ):
        self.op = op
        self.better = better
        self.callback = callback
        self.n_generations = n_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.verbose = verbose
        self.return_history = return_history

    def __call__(self):
        if self.return_history:
            self.history = {"p_front": [], "p_front_fitnesses": []}

        # Create initial population and evaluate
        population = self.create_population()
        fitnesses_population = self.evaluate_population(population)

        search_state = (population, 0)

        if self.verbose or self.return_history:
            p_front, p_front_fitnesses = self.calculate_pareto_front(
                population, fitnesses_population
            )
        if self.verbose:
            print(
                f"Generation 0: Pareto Front Length: {len(p_front)}, Pareto Front Pheno: {p_front}, Pareto Front Fitness: {p_front_fitnesses}\n"
            )

        if self.return_history:
            self.history["p_front"].append(p_front)
            self.history["p_front_fitnesses"].append(p_front_fitnesses)
        if self.callback:
            self.callback(search_state)

        for gen in range(self.n_generations):
            # Variation operators: crossover and mutation
            offspring = self.create_offspring(population)
            fitnesses_offspring = self.evaluate_population(offspring)

            # Combine parents and offspring
            population += offspring
            fitnesses_population += fitnesses_offspring

            # Non-dominated sorting to rank population
            fronts = self.fast_non_dominated_sort(population, fitnesses_population)

            # Select new population using fronts and crowding distance
            population, fitnesses_population = self.selection(
                population, fitnesses_population, fronts
            )

            search_state = (population, gen + 1)

            if self.verbose or self.return_history:
                p_front, p_front_fitnesses = self.calculate_pareto_front(
                    population, fitnesses_population
                )
            if self.verbose and (gen + 1) % 10 == 0:
                print(
                    f"Generation {gen + 1}: Pareto Front Length: {len(p_front)}, Pareto Front Pheno: {p_front}, Pareto Front Fitness: {p_front_fitnesses}\n"
                )

            if self.return_history:
                self.history["p_front"].append(p_front)
                self.history["p_front_fitnesses"].append(p_front_fitnesses)

            if self.callback and self.callback(search_state):
                break

        # Final front
        p_front, p_front_fitnesses = self.calculate_pareto_front(
            population, fitnesses_population
        )
        if self.return_history:
            return p_front, population, self.history
        else:
            return p_front, population

    def create_population(self):
        return [self.op.create_ind() for _ in range(self.population_size)]

    def evaluate_population(self, population):
        return [self.op.evaluate_ind(ind) for ind in population]

    def dominates(self, fitness1, fitness2):
        # Pareto dominance definition based on the optimization direction
        if self.better == min:
            return all(f1 <= f2 for f1, f2 in zip(fitness1, fitness2)) and any(
                f1 < f2 for f1, f2 in zip(fitness1, fitness2)
            )
        elif self.better == max:
            return all(f1 >= f2 for f1, f2 in zip(fitness1, fitness2)) and any(
                f1 > f2 for f1, f2 in zip(fitness1, fitness2)
            )

    def create_offspring(self, population):
        # Apply crossover or copy
        offspring_crossover = [
            (
                self.op.crossover_ind(
                    random.choice(population), random.choice(population)
                )
                if random.random() < self.crossover_rate
                else random.choice(population)
            )
            for _ in range(self.population_size)
        ]

        # Apply mutation
        offspring_final = [
            (
                self.op.mutate_point_ind(offspring_crossover[i])
                if random.random() < self.mutation_rate
                else offspring_crossover[i]
            )
            for i in range(len(offspring_crossover))
        ]

        return offspring_final

    def fast_non_dominated_sort(self, population, fitnesses):
        domination_count = [0] * len(population)
        dominated_solutions = [[] for _ in range(len(population))]
        fronts = [[]]

        for i, fit_i in enumerate(fitnesses):
            for j, fit_j in enumerate(fitnesses):
                if self.dominates(fit_i, fit_j):
                    dominated_solutions[i].append(j)
                elif self.dominates(fit_j, fit_i):
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def selection(self, population, fitnesses, fronts):
        new_population = []
        new_fitnesses = []

        for front in fronts:
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend([population[i] for i in front])
                new_fitnesses.extend([fitnesses[i] for i in front])
            else:
                crowding_distances = self.calculate_crowding_distances(
                    [fitnesses[i] for i in front]
                )
                sorted_indices = sorted(
                    range(len(front)),
                    key=lambda i: crowding_distances[i],
                    reverse=True,
                )
                remaining = self.population_size - len(new_population)
                selected = [front[i] for i in sorted_indices[:remaining]]
                new_population.extend([population[i] for i in selected])
                new_fitnesses.extend([fitnesses[i] for i in selected])
                break

        return new_population, new_fitnesses

    def calculate_crowding_distances(self, front):
        distances = [0] * len(front)
        n_objectives = len(front[0])

        for m in range(n_objectives):
            sorted_indices = sorted(range(len(front)), key=lambda i: front[i][m])
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float("inf")
            for i in range(1, len(front) - 1):
                distances[sorted_indices[i]] += (
                    front[sorted_indices[i + 1]][m] - front[sorted_indices[i - 1]][m]
                )

        return distances

    def calculate_pareto_front(self, population, fitnesses_population):
        p_front_indices = self.fast_non_dominated_sort(
            population, fitnesses_population
        )[0]

        # Use a set to store unique (phenotype, fitness) pairs
        unique_pareto = set()

        for i in p_front_indices:
            pheno = tuple(getattr(population[i], "pheno", population[i]))
            fitness = tuple(fitnesses_population[i])

            # Add the (phenotype, fitness) pair to the set
            unique_pareto.add((pheno, fitness))

        # Unpack the unique Pareto front
        p_front = [list(pheno) for pheno, _ in unique_pareto]
        p_front_fitnesses = [fitness for _, fitness in unique_pareto]

        return p_front, p_front_fitnesses


# TEST ON BI-OBJECTIVE OneMinMax


class BinaryProblem:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions

    def create_ind(self):
        return [random.randint(0, 1) for _ in range(self.n_dimensions)]

    def evaluate_ind(self, individual):
        pheno = getattr(individual, "pheno", individual)
        one_max = sum(pheno)  # Number of ones
        one_min = len(pheno) - one_max  # Number of zeros
        return one_max, one_min

    def crossover_ind(self, parent1, parent2):
        crossover_point = random.randint(1, self.n_dimensions - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return random.choice([child1, child2])

    def mutate_point_ind(self, individual):
        return [1 - bit if random.random() < 0.01 else bit for bit in individual]


# Example usage:
if __name__ == "__main__":
    problem = BinaryProblem(n_dimensions=5)
    NSGA_II = NSGAII(
        problem,
        n_generations=50,
        population_size=10,
        verbose=False,
        return_history=False,
    )
    pareto_front, population = NSGA_II()

    print(f"Pareto front size: {len(pareto_front)}\n")
    print(f"Pareto front phenotype: {pareto_front}\n")

    pareto_front_fitnesses = []
    for i in range(len(pareto_front)):
        pareto_front_fitnesses.append(BinaryProblem.evaluate_ind(None, pareto_front[i]))
    print(f"Pareto front fitnesses {pareto_front_fitnesses}")
