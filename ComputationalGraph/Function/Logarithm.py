import math
from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class Logarithm(Function):
    """
    Applies the natural logarithm function to a tensor.
    """

    def calculate(self, value: Tensor) -> Tensor:
        """
        Applies the natural logarithm to each element of the input tensor.
        """

        new_data = [math.log(x) for x in value.getData()]

        return Tensor(new_data, value.getShape())

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the Logarithm function.
        Note: 'value' here is the output of the log function, so x = exp(value).
        """
        new_data = []

        for val, back_val in zip(value.getData(), backward.getData()):
            derivative_val = 1.0 / math.exp(val)
            new_data.append(derivative_val * back_val)

        return Tensor(new_data, value.getShape())

    def addEdge(self, input_nodes: List[ComputationalNode], is_biased: bool) -> ComputationalNode:
        """
        Adds a Logarithm node to the graph.
        """
        new_node = FunctionNode(function=self, is_biased=is_biased)
        input_nodes[0].add(new_node)
        return new_node