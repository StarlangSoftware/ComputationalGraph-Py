import random
from abc import ABC, abstractmethod
from typing import List

class Initialization(ABC):
    """
    Interface for weight initialization strategies in the computational graph.
    """

    @abstractmethod
    def initialize(self, row: int, column: int, rng: random.Random) -> List[float]:
        """
        Initializes a list of weights based on the specified dimensions.

        :param row: The number of rows for the weight matrix.
        :param column: The number of columns for the weight matrix.
        :param rng: A random number generator instance.
        :return: A flat list of initialized float values.
        """
        pass