import math
import random
from typing import List

from .Initialization import Initialization

class HeUniformInitialization(Initialization):
    """
    He Uniform Initialization.

    This method initializes weights using a uniform distribution, which is typically
    optimized for layers with ReLU activation functions. It helps in maintaining
    the variance of activations throughout the network layers.
    """

    def initialize(self, row: int, column: int, rng: random.Random) -> List[float]:
        """
        Generates the initialized weight values.

        :param row: The number of rows in the matrix (typically represents the output size / number of neurons).
        :param column: The number of columns in the matrix (typically represents the input size / fan-in).
        :param rng: A random number generator instance for reproducibility.
        :return: A flat list of floats containing the initialized weight values.
        """
        sqrt_col = math.sqrt(6.0 / column)
        sqrt_row = math.sqrt(6.0 / row)

        multiplier = sqrt_col + sqrt_row
        offset = sqrt_row

        return [(multiplier * rng.random()) - offset for _ in range(row * column)]