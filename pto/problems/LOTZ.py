##### Leading Ones and Trailing Zeros (LOTZ)

from pto import run, rnd

#################
# INSTANCE DATA #
#################

size = 5  # Problem size (length of binary vector)
better = max  # Optimization goal (minimizing or maximizing)

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
    1. Count the number of 1s from the start of a solution (Stop counting at first 0).
    1. Count the number of 0s from the end of a solution (Stop counting at first 1).

    Args:
        solution (list): Binary vector solution.

    Returns:
        tuple: (Number of leading 1s, Number of trailing 0s)
    """
    # Leading ones
    try:
        num_leading_ones = solution.index(0)
    except ValueError:
        # No zeros found: all ones
        num_leading_ones = len(solution)

    # Trailing zeros
    try:
        num_trailing_zeros = solution[::-1].index(1)
    except ValueError:
        # No ones found: all zeros
        num_trailing_zeros = len(solution)

    return num_leading_ones, num_trailing_zeros


########################
# RUN OPTIMIZATION     #
########################

if __name__ == "__main__":
    result = run(
        generator,
        fitness,
        gen_args=(size,),
        Solver="NSGAII",
        solver_args={
            "n_generations": 50,
            "population_size": 10,
            "verbose": False,
            "return_history": False,
        },
        better=better,
    )

    pareto_front, population = result

    print(f"Pareto front size: {len(pareto_front)}\n")
    print(f"Pareto front solutions (phenotype): {pareto_front}\n")

    pareto_front_fitnesses = [fitness(sol) for sol in pareto_front]
    print(f"Pareto front fitnesses: {pareto_front_fitnesses}")
