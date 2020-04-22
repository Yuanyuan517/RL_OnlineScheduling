import numpy as np


# generate a set of seeds with size = size of time steps
def generate_random_seeds(initial_seed, size):
    np.random.seed(initial_seed)
    return np.random.random_integers(0, 100000, size)
