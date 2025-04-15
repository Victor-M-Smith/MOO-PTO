from pto import run, rnd
import numpy as np

#################
# INSTANCE DATA #
#################

size = 5  # Total size of the solution vector
better = min  # Optimization goal (minimizing or maximizing)

# Define search space
lower_search_bound = -10  # Minimum value for each element in solution vector.
upper_search_bound = 10  # Maximum value for each element in solution vector.

# Define fixed centers for the spheres
sphere_1_center = np.array([-5.0] * size)  # Center of Sphere 1
sphere_2_center = np.array([5.0] * size)  # Center of Sphere 2

######################
# SOLUTION GENERATOR #
######################


def generator(size, lower_bound=-100, upper_bound=100):
    """
    Generates a random vector of floats within the given bounds.

    Args:
        size (int): Length of the solution vector.
        lower_bound (float): Minimum value for each element.
        upper_bound (float): Maximum value for each element.

    Returns:
        list: A randomly generated solution vector.
    """
    return [rnd.uniform(lower_bound, upper_bound) for _ in range(size)]


####################
# FITNESS FUNCTION #
####################


def fitness(solution, sphere_1_center, sphere_2_center):
    """
    Computes the bi-objective fitness values for the solution.

    The objectives are the squared Euclidean distances to two fixed points (centers of the spheres).

    Args:
        solution (list): The solution vector.

    Returns:
        tuple: The fitness values for both objectives (distance to each center).
    """
    sphere1 = np.sqrt(sum((solution - sphere_1_center) ** 2))
    sphere2 = np.sqrt(sum((solution - sphere_2_center) ** 2))
    return sphere1, sphere2


####################
# RUN OPTIMIZATION #
####################

if __name__ == "__main__":
    # Run the optimization Algorithm
    result = run(
        generator,
        fitness,
        fit_args=(sphere_1_center, sphere_2_center),
        gen_args=(size, lower_search_bound, upper_search_bound),
        Solver="NSGAII",
        solver_args={
            "n_generations": 50,
            "population_size": 10,
            "verbose": False,
            "return_history": False,
        },
        better=better,
    )

    # print("Raw result:", result)
    pareto_front, population = result

    print(f"Pareto front size: {len(pareto_front)}\n")
    print(f"Pareto front solutions (phenotype): {pareto_front}\n")

    pareto_front_fitnesses = []
    for i in range(len(pareto_front)):
        pareto_front_fitnesses.append(
            fitness(pareto_front[i], sphere_1_center, sphere_2_center)
        )

    print(f"Pareto front fitnesses: {pareto_front_fitnesses}")
