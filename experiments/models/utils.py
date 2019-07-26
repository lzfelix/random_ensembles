import numpy as np
from opytimizer import Opytimizer
from opytimizer.spaces.search import SearchSpace
from opytimizer.core.function import Function


def optimize(metaheuristic,
             target,
             n_agents,
             n_variables,
             n_iterations,
             lb,
             ub,
             hyperparams):
    """Abstract all the Opytimizer mechanics into a single method."""
    space = SearchSpace(n_agents, n_variables, n_iterations, lb, ub)
    optimizer = metaheuristic(hyperparams=hyperparams)

    task = Opytimizer(space, optimizer, Function(target))
    return task.start()


def get_top_models(history, n_agents):
    """Gets all the best models found during optimization."""

    best_indices = list()
    for iter_no, iteration_best in enumerate(history.best_agent):
        # Finding the actual index
        position, fitness, index = iteration_best
        cumulative_index = index + iter_no * n_agents

        # Opytimizer discards the current iteration best if it is
        # worse than the previous last solution. Check for that so
        # each model is selected only once.
        # Despite the name, history.agents contains all the agents
        # values for the i-th iteration
        actual_fitness = history.agents[iter_no][index][-1]
        if np.isclose(actual_fitness, fitness) or iter_no == 0:
            best_indices.append((index, cumulative_index, fitness))

    return best_indices
