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

    __p: float
    __random: random.Random
    __mask: List[float]

    def __init__(self, p: float, rng: Optional[random.Random] = None) -> None:
        """
        Initializes the Dropout function.

        :param p: The probability of dropping an element.
        :param rng: A random.Random instance.
        """
        self.__p = p
        self.__random = rng if rng is not None else random.Random()
        self.__mask = []

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes dropout values and stores the mask for the backward pass.

        :param value: Input tensor.
        :return: Tensor after dropout is applied.
        """
        self.__mask.clear()
        multiplier = 1.0 / (1.0 - self.__p)
        new_data = []

        old_values = value.getData()
        for old_value in old_values:
            r = self.__random.random()
            if r > self.__p:
                self.__mask.append(multiplier)
                new_data.append(old_value * multiplier)
            else:
                self.__mask.append(0.0)
                new_data.append(0.0)

        return Tensor(new_data, value.getShape())

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Calculates the derivative of dropout using the stored mask.

        :param value: Current tensor value.
        :param backward: Backward gradient tensor.
        :return: Resulting gradient tensor.
        """
        mask_tensor = Tensor(self.__mask, value.getShape())
        return backward.hadamardProduct(mask_tensor)

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        """
        Adds a Dropout node to the computational graph.

        :param input_nodes: Input computational nodes.
        :param is_biased: Indicates whether the edge is biased.
        :return: Newly created computational node.
        """
        new_node = FunctionNode(function=self, is_biased=is_biased)
        input_nodes[0].add(new_node)
        return new_node