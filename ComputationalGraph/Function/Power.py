import math
from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class Power(Function):
    """
    Power function representation for the computational graph.
    Computes x^n for a given tensor.
    """

    __n: int

    def __init__(self, n: int = 2) -> None:
        """
        Initializes the Power function.

        :param n: The exponent.
        """
        self.__n = n

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes the Power of the given tensor.

        :param value: The tensor whose values are to be computed.
        :return: A new Tensor containing pow(x, n).
        """
        new_data = [math.pow(val, self.__n) for val in value.getData()]

        return Tensor(new_data, value.getShape())

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the Power function.

        :param value: The output tensor of the Power(x) operation.
        :param backward: The backward gradient tensor.
        :return: A new Tensor containing the gradient values.
        """
        new_data = []

        for val, backward_value in zip(value.getData(), backward.getData()):
            derivative_val = self.__n * math.pow(math.pow(val, 1.0 / self.__n), self.__n - 1)
            new_data.append(derivative_val * backward_value)

        return Tensor(new_data, value.getShape())

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        """
        Adds a Power node to the computational graph.

        :param input_nodes: Input computational nodes.
        :param is_biased: Indicates whether the edge is biased.
        :return: Newly created computational node.
        """
        new_node = FunctionNode(function=self, is_biased=is_biased)
        input_nodes[0].add(new_node)
        return new_node