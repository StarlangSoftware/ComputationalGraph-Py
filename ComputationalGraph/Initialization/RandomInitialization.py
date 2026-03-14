import random
from typing import List

from .Initialization import Initialization

class RandomInitialization(Initialization):
    """
    Random Uniform Initialization.

    This method initializes the weights with small random values uniformly distributed
    between -0.01 and 0.01. This is a basic initialization strategy used to break
    symmetry between neurons.
    """

    def initialize(self, row: int, column: int, rng: random.Random) -> List[float]:
        """
        Generates the initialized weight values.

        :param row: The number of rows in the matrix.
        :param column: The number of columns in the matrix.
        :param rng: A random number generator instance for reproducibility.
        :return: A flat list of floats containing the initialized weight values.
        """

        return [-0.01 + (0.02 * rng.random()) for _ in range(row * column)]