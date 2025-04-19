from pto import run, rnd

#################
# INSTANCE DATA #
#################

size = 5  # Total size of the solution vector
better = min  # Optimization goal (minimizing or maximizing)

# Define search space
lower_search_bound = -10  # Minimum value for each element in solution vector.
upper_search_bound = 10  # Maximum value for each element in solution vector.

# Define fixed centers for the spheres
sphere_1_center = [-5.0] * size  # Center of Sphere 1
sphere_2_center = [5.0] * size  # Center of Sphere 2

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
    sphere1 = sum((solution[i] - sphere_1_center[i]) ** 2 for i in range(size)) ** 0.5
    sphere2 = sum((solution[i] - sphere_2_center[i]) ** 2 for i in range(size)) ** 0.5

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
            "return_history": True,
        },
        better=better,
    )

    # print("Raw result:", result)
    pareto_front_pheno, pareto_front_geno, history = result

    print(f"Pareto front size: {len(pareto_front_pheno)}\n")
    print(f"Pareto front solutions (phenotype): {pareto_front_pheno}\n")
    print(f"Pareto front fitnesses {history['p_front_fitnesses'][-1]}")
