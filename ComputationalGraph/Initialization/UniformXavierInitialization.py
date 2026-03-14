import math
import random
from typing import List

from .Initialization import Initialization

class UniformXavierInitialization(Initialization):
    """
    Xavier Uniform Initialization.

    This method initializes weights using a uniform distribution within the range
    [-limit, limit], where the limit is sqrt(6 / (fan_in + fan_out)).
    This strategy is designed to keep the scale of the gradients roughly the same
    in all layers and is commonly used with Sigmoid or Tanh activation functions.
    """

    def initialize(self, row: int, column: int, rng: random.Random) -> List[float]:
        """
        Generates the initialized weight values.

        :param row: The number of rows in the matrix (typically represents fan-out / output size).
        :param column: The number of columns in the matrix (typically represents fan-in / input size).
        :param rng: A random number generator instance for reproducibility.
        :return: A flat list of floats containing the initialized weight values.
        """

        limit = math.sqrt(6.0 / (row + column))

        return [(2.0 * rng.random() - 1.0) * limit for _ in range(row * column)]