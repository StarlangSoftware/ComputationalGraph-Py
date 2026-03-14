import math
from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor

class Tanh(Function):
    """
    Hyperbolic Tangent (Tanh) activation function representation for the computational graph.
    """

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes the Tanh activation for the given tensor.

        :param value: The tensor whose values are to be computed.
        :return: A new Tensor containing Tanh(x).
        """
        new_data = [math.tanh(val) for val in value.getData()]

        return Tensor(new_data, value.getShape())

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the Tanh activation function.
        Note: 'value' here is the output of the Tanh(x) forward pass (y).
        The derivative is (1 - y^2).

        :param value: The output tensor of the Tanh(x) operation.
        :param backward: The backward gradient tensor.
        :return: A new Tensor containing the gradient values.
        """
        new_data = [
            (1.0 - val * val) * back_val
            for val, back_val in zip(value.getData(), backward.getData())
        ]

        return Tensor(new_data, value.getShape())

    def addEdge(self, input_nodes: List[ComputationalNode], is_biased: bool) -> ComputationalNode:
        """
        Adds a Tanh node to the computational graph.
        """
        new_node = FunctionNode(function=self, is_biased=is_biased)
        input_nodes[0].add(new_node)
        return new_node