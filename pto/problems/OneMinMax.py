##### OneMinMax

from pto import run, rnd

#################
# INSTANCE DATA #
#################

size = 5  # Problem size (length of binary vector)
better = min  # Optimization goal (minimizing or maximizing)

#######################
# SOLUTION GENERATOR  #
#######################


def generator(size):
    """
    Generates a random binary vector of given size.

    Args:
        size (int): Length of the binary vector.

    Returns:
        list: Random vector containing 0s and 1s.
    """
    return [rnd.choice([0, 1]) for _ in range(size)]


####################
# FITNESS FUNCTION #
####################


def fitness(solution):
    """
    Bi-objective fitness function.

    Objectives:
    1. Count of 1s in the solution.
    2. Count of 0s in the solution.

    Args:
        solution (list): Binary vector solution.

    Returns:
        tuple: (Number of 1s, Number of 0s)
    """
    num_ones = sum(solution)
    num_zeros = len(solution) - num_ones
    return num_ones, num_zeros


####################
# RUN OPTIMIZATION #
####################

if __name__ == "__main__":
    # Run the optimization algorithm
    result = run(
        generator,
        fitness,
        gen_args=(size,),
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
