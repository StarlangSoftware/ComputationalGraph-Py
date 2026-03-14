import math
from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class Sigmoid(Function):
    """
    Sigmoid activation function representation for the computational graph.
    """

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes the Sigmoid activation for the given tensor.

        :param value: The tensor whose values are to be computed.
        :return: A new Tensor containing Sigmoid(x) = 1 / (1 + exp(-x)).
        """
        new_data = [1.0 / (1.0 + math.exp(-val)) for val in value.getData()]

        return Tensor(new_data, value.getShape())

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the Sigmoid activation function.
        Note: 'value' here is the output of the Sigmoid(x) forward pass (let's call it y).
        The derivative of sigmoid is y * (1 - y).

        :param value: The output tensor of the Sigmoid(x) operation.
        :param backward: The backward gradient tensor.
        :return: A new Tensor containing the gradient values.
        """
        new_data = [
            val * (1.0 - val) * back_val
            for val, back_val in zip(value.getData(), backward.getData())
        ]

        return Tensor(new_data, value.getShape())

    def addEdge(self, input_nodes: List[ComputationalNode], is_biased: bool) -> ComputationalNode:
        """
        Adds a Sigmoid node to the computational graph.
        """
        new_node = FunctionNode(function=self, is_biased=is_biased)
        input_nodes[0].add(new_node)
        return new_node