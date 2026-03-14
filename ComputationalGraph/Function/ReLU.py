from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class ReLU(Function):
    """
    Rectified Linear Unit (ReLU) activation function representation for the computational graph.
    """

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes the ReLU activation for the given tensor.

        :param value: The tensor whose values are to be computed.
        :return: A new Tensor containing ReLU(x) = max(x, 0).
        """
        new_data = [max(val, 0.0) for val in value.getData()]

        return Tensor(new_data, value.getShape())

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the ReLU activation function.
        Note: 'value' here is the output of the ReLU(x) forward pass.

        :param value: The output tensor of the ReLU(x) operation.
        :param backward: The backward gradient tensor.
        :return: A new Tensor containing the gradient values.
        """
        new_data = [
            back_val if val > 0 else 0.0
            for val, back_val in zip(value.getData(), backward.getData())
        ]

        return Tensor(new_data, value.getShape())

    def addEdge(self, input_nodes: List[ComputationalNode], is_biased: bool) -> ComputationalNode:
        """
        Adds a ReLU node to the computational graph.
        """
        new_node = FunctionNode(function=self, is_biased=is_biased)
        input_nodes[0].add(new_node)
        return new_node