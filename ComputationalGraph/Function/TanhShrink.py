from typing import List

from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from .Tanh import Tanh
from .Negation import Negation

class TanhShrink(Tanh):
    """
    TanhShrink activation function representation for the computational graph.
    Inherits from Tanh to compute x - Tanh(x).
    """

    def addEdge(self, input_nodes: List[ComputationalNode], is_biased: bool) -> ComputationalNode:
        """
        Constructs the subgraph for the TanhShrink activation operation.

        :param input_nodes: A list of input computational nodes.
        :param is_biased: Indicates whether the connection is biased.
        :return: The final node representing x - Tanh(x).
        """
        tanh = FunctionNode(function=self, is_biased=False)
        input_nodes[0].add(tanh)

        negative_tanh = FunctionNode(function=Negation(), is_biased=False)
        tanh.add(negative_tanh)

        tanh_shrink = ComputationalNode(learnable=False, is_biased=is_biased)

        input_nodes[0].add(tanh_shrink)

        negative_tanh.add(tanh_shrink)

        return tanh_shrink