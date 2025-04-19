###### motsp

from pto import run, rnd
import random

#################
# INSTANCE DATA #
#################

size = 5  # Number of cities in MOTSP problem
better = min  # Optimization goal (minimizing or maximizing)


# Generate distance and time matrices (nxn matracies with n=size)
def generate_problem_data(size, random_state=None):
    if random_state is not None:
        # Set the random seed for reproducibility of the generated matrices
        random.seed(random_state)

    distance_matrix = [
        [round(random.random(), 2) for _ in range(size)] for _ in range(size)
    ]
    time_matrix = [
        [round(random.random(), 2) for _ in range(size)] for _ in range(size)
    ]

    if random_state is not None:
        # Reset the random seed to avoid affecting downstream randomness
        random.seed(None)

    return distance_matrix, time_matrix


# Generate distance and time matrices
distance_matrix, time_matrix = generate_problem_data(size, 0)

#####################
# SOLUTION GENERATOR #
#####################


def generator(size):
    """
    Generates a random tour (permutation of cities).

    Args:
        size (int): Number of cities.

    Returns:
        list: A randomly shuffled list representing the tour.
    """
    cities_list = list(range(size))
    rnd.shuffle(cities_list)
    return cities_list


####################
# FITNESS FUNCTION #
####################


def fitness(tour, size, dist_matrix, time_matrix):
    """
    Computes the bi-objective fitness values for a given tour.

    Args:
        tour (list): List of city indices representing the tour.
        size (int): Number of cities.
        dist_matrix (list of lists): Distance matrix.
        time_matrix (list of lists): Time matrix.

    Returns:
        tuple: Total distance and total time for the given tour.
    """
    total_distance = sum(
        dist_matrix[tour[i]][tour[(i + 1) % size]] for i in range(size)
    )
    total_time = sum(time_matrix[tour[i]][tour[(i + 1) % size]] for i in range(size))
    return total_distance, total_time


######################
# MAIN EXECUTION #
######################

if __name__ == "__main__":
    result = run(
        generator,
        fitness,
        gen_args=(size,),
        fit_args=(size, distance_matrix, time_matrix),
        Solver="NSGAII",
        solver_args={
            "n_generations": 50,
            "population_size": 10,
            "verbose": False,
            "return_history": True,
        },
        better=better,
    )

    # print("Raw result:", result)
    pareto_front_pheno, pareto_front_geno, history = result

    print(f"Pareto front size: {len(pareto_front_pheno)}\n")
    print(f"Pareto front solutions (phenotype): {pareto_front_pheno}\n")
    print(f"Pareto front fitnesses {history['p_front_fitnesses'][-1]}")
