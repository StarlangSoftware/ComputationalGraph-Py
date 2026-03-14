from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ComputationalGraph.Node.ComputationalNode import ComputationalNode
    from Math.Tensor import Tensor


class Function(ABC):
    """
    An interface representing a mathematical function in the computational graph.
    """

    @abstractmethod
    def calculate(self, matrix: 'Tensor') -> 'Tensor':
        """
        Calculates the forward pass of the function.

        :param matrix: The input tensor.
        :return: The resulting tensor after applying the function.
        """
        pass

    @abstractmethod
    def derivative(self, value: 'Tensor', backward: 'Tensor') -> 'Tensor':
        """
        Calculates the derivative (backward pass) of the function.

        :param value: The current tensor value.
        :param backward: The backward gradient tensor.
        :return: The resulting gradient tensor.
        """
        pass

    @abstractmethod
    def addEdge(self, input_nodes: List['ComputationalNode'], is_biased: bool) -> 'ComputationalNode':
        """
        Adds an edge to the computational graph.

        :param input_nodes: A list of input computational nodes.
        :param is_biased: Indicates whether the connection is biased.
        :return: The resulting computational node.
        """
        pass
