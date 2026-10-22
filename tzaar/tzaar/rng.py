"""One shared random number generator, so runs are repeatable."""
import numpy as np

rng = np.random.default_rng(0)
