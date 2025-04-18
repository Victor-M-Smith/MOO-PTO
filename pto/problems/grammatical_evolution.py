##### GRAMMATICAL EVOLUTION

from pto import run, rnd

import random  # for generating the problem data
import numpy as np

np.seterr(all="raise")


#################
# INSTANCE DATA #
#################


### GRAMMAR

n_vars = 3
grammar = {
    "<expr>": [["(", "<expr>", "<biop>", "<expr>", ")"], ["<var>"], ["<const>"]],
    "<biop>": [["+"], ["-"], ["*"], ["/"]],
    "<var>": [["x", "[:,", "<varidx>", "]"]],
    "<varidx>": [
        [str(i)] for i in range(n_vars)
    ],  # over-write this later for a different value of n_vars
    "<const>": [
        ["0.0"],
        ["0.1"],
        ["0.2"],
        ["0.3"],
        ["0.4"],
        ["0.5"],
        ["0.6"],
        ["0.7"],
        ["0.8"],
        ["0.9"],
        ["1.0"],
    ],
}

### TRAINING SET


def make_training_data(n_samples, n_vars):
    coef = [0.5 - random.random() for _ in range(n_vars)]
    target = lambda x: sum(
        coef[i] * (x[i] ** i * x[(i + 1) % n_vars] * x[(i + 2) % n_vars])
        for i in range(n_vars)
    )  # target function
    X = np.array(
        [[2 * random.random() for _ in range(n_vars)] for _ in range(n_samples)]
    )  # training inputs
    y = np.array([target(xi) for xi in X])  # training outputs
    return X, y


better = min


######################
# SOLUTION GENERATOR #
######################


def generator(grammar, max_depth=10):

    def expand(left_sym, depth):
        if left_sym in grammar:
            if depth <= 1:
                rules = [
                    rule for rule in grammar[left_sym] if "<expr>" not in rule
                ]  # not recursive
            else:
                rules = grammar[left_sym]
            selected_index = rnd.choice(
                range(len(rules))
            )  # rnd.randint(0, len(rules)-1)
            return "".join(
                [
                    expand(right_sym, depth - 1)
                    for i, right_sym in enumerate(rules[selected_index])
                ]
            )
        else:
            return left_sym

    return expand("<expr>", 10)  # depth-first expansion


####################
# FITNESS FUNCTION #
####################


# fitness
def fitness(expr, X, y):

    # string to function
    f = eval("lambda x : " + expr)

    MAX_ERR = 20

    try:
        yhat = f(X)
        err = np.linalg.norm(y - yhat)
    except:
        err = MAX_ERR * len(X)

    # try:
    #     # predictions on training set
    #     yhat = list(map(f, X))
    # except:
    #     # give a large error, if an evaluation error occurs (e.g., division by 0)
    #     err = MAX_ERR
    # else:
    #     # error on training set (when no evaluation error)
    #     err = min(MAX_ERR, sum(abs(yi - yhati) for yi, yhati in zip(y, yhat)))

    return err


if __name__ == "__main__":
    n_vars = 9
    n_samples = n_vars * 10
    grammar["<varidx>"] = [[str(i)] for i in range(n_vars)]
    X_train, y_train = make_training_data(n_samples, n_vars)
    (pheno, geno), fx = run(
        generator,
        fitness,
        gen_args=(grammar,),
        fit_args=(X_train, y_train),
        solver_args={"n_generation": 10000},
        better=better,
    )
    print(f"Solution {pheno}")
    print(f"Fitness {fx}")
