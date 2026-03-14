import random
from typing import List, Optional

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class Dropout(Function):
    """
    Implements the Dropout function for regularization in the computational graph.
    """

    def __init__(self, p: float, rng: Optional[random.Random] = None):
        """
        Initializes the Dropout function.

        :param p: The probability of dropping an element.
        :param rng: A random.Random instance (equivalent to Java's Random).
        """
        self.p: float = p
        # Use the provided RNG or the global random module
        self.random: random.Random = rng if rng is not None else random.Random()
        self.mask: List[float] = []

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes the dropout values and stores the mask for the backward pass.
        """
        self.mask.clear()
        multiplier = 1.0 / (1.0 - self.p)
        new_data = []

        old_values = value.getData()
        for old_value in old_values:
            r = self.random.random()
            if r > self.p:
                self.mask.append(multiplier)
                new_data.append(old_value * multiplier)
            else:
                self.mask.append(0.0)
                new_data.append(0.0)

        return Tensor(new_data, value.getShape())

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Calculates the derivative of the dropout using the stored mask.
        """
        mask_tensor = Tensor(self.mask, value.getShape())
        return backward.hadamardProduct(mask_tensor)

    def addEdge(self, input_nodes: List[ComputationalNode], is_biased: bool) -> ComputationalNode:
        """
        Adds a Dropout node to the computational graph.
        """
        new_node = FunctionNode(function=self, is_biased=is_biased)
        input_nodes[0].add(new_node)
        return new_node