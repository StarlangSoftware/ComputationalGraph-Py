import math
from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class DELU(Function):
    """
    DELU activation function representation for the computational graph.
    """

    def __init__(self, a: float = 1.0, b: float = 2.0, xc: float = 1.25643):
        """
        Initializes the DELU activation function.
        Consolidates both Java constructors using default keyword arguments.

        :param a: Parameter 'a' for the DELU function.
        :param b: Parameter 'b' for the DELU function.
        :param xc: Threshold parameter 'xc' for the DELU function.
        """
        self.a: float = a
        self.b: float = b
        self.xc: float = xc

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes the DELU activation for the given value tensor.

        :param value: The tensor whose values are to be computed.
        :return: A new Tensor containing DELU(x).
        """
        new_data = []

        for val in value.getData():
            if val > self.xc:
                new_data.append(val)
            else:
                new_data.append((math.exp(self.a * val) - 1) / self.b)

        return Tensor(new_data, value.getShape())

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the DELU activation function.

        :param value: The output tensor of the DELU(x) operation.
        :param backward: The backward gradient tensor.
        :return: A new Tensor containing the gradient values.
        """
        new_data = []

        for val, back_val in zip(value.getData(), backward.getData()):
            if val > self.xc:
                new_data.append(back_val)
            else:
                derivative_val = (val * self.b + 1) * (self.a / self.b)
                new_data.append(back_val * derivative_val)

        return Tensor(new_data, value.getShape())


    def addEdge(self, input_nodes: List[ComputationalNode], is_biased: bool) -> ComputationalNode:
        """
        Adds a DELU node to the computational graph.
        """
        new_node = FunctionNode(function=self, is_biased=is_biased)
        input_nodes[0].add(new_node)
        return new_node