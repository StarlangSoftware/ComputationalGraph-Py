import math
from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor

class ELU(Function):
    """
    Exponential Linear Unit (ELU) activation function representation for the computational graph.
    """

    def __init__(self, a: float = 1.0):
        """
        Initializes the ELU activation function.
        Consolidates the Java constructors using a default keyword argument.

        :param a: The alpha parameter for the ELU function (defaults to 1.0).
        """
        self.a: float = a

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes the ELU activation for the given tensor.

        :param value: The tensor whose values are to be computed.
        :return: A new Tensor containing ELU(x).
        """
        new_data = []
        for old_value in value.getData():
            if old_value < 0:
                new_data.append(self.a * (math.exp(old_value) - 1.0))
            else:
                new_data.append(old_value)

        return Tensor(new_data, value.getShape())

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the ELU activation function.

        :param value: The output tensor of the ELU(x) operation.
        :param backward: The backward gradient tensor.
        :return: A new Tensor containing the gradient values.
        """
        new_data = []

        for old_value, backward_value in zip(value.getData(), backward.getData()):
            # Because old_value is the output (y) of the ELU function,
            # if x < 0, then y < 0, and dy/dx = y + a
            if old_value < 0:
                new_data.append((old_value + self.a) * backward_value)
            else:
                new_data.append(backward_value)

        return Tensor(new_data, value.getShape())

    def addEdge(self, input_nodes: List[ComputationalNode], is_biased: bool) -> ComputationalNode:
        """
        Adds an ELU node to the computational graph.
        """
        new_node = FunctionNode(function=self, is_biased=is_biased)
        input_nodes[0].add(new_node)
        return new_node