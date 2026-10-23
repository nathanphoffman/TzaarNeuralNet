"""One shared random number generator, so runs are repeatable."""
import numpy as np

random_generator = np.random.default_rng(0)
