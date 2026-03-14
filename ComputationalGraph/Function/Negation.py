from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor

class Negation(Function):
    """
    Negation function representation for the computational graph.
    """

    def calculate(self, value: Tensor) -> Tensor:
        """
        Negates the values of the given tensor.

        :param value: The tensor whose values are to be negated.
        :return: A new Tensor with negated values.
        """
        new_data = [-old_value for old_value in value.getData()]

        return Tensor(new_data, value.getShape())

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Calculates the derivative of the Negation function.
        Since f(x) = -x, f'(x) = -1. The gradient is simply -backward.

        :param value: The output tensor of the Negation function (unused here).
        :param backward: The backward gradient tensor.
        :return: A new Tensor containing the negated backward gradients.
        """
        # A simple list comprehension to negate all backward elements
        new_data = [-backward_value for backward_value in backward.getData()]

        return Tensor(new_data, backward.getShape())

    def addEdge(self, input_nodes: List[ComputationalNode], is_biased: bool) -> ComputationalNode:
        """
        Adds a Negation node to the computational graph.
        """
        new_node = FunctionNode(function=self, is_biased=is_biased)
        input_nodes[0].add(new_node)
        return new_node