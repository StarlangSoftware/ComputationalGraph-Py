from typing import List

from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode
from ComputationalGraph.Function.Logarithm import Logarithm


class CrossEntropyLoss(Logarithm):
    """
    Cross Entropy Loss function representation for the computational graph.
    Inherits the calculate and derivative math from Logarithm, but overrides the graph wiring.
    """

    def addEdge(self, input_nodes: List[ComputationalNode], is_biased: bool) -> ComputationalNode:
        """
        Constructs the subgraph for the cross-entropy loss operation.
        """
        # Create a function node that uses the Logarithm math (since 'self' is a Logarithm)
        logy = FunctionNode(function=self, is_biased=False)
        input_nodes[0].add(logy)

        # Create a Hadamard multiplication node for y * log(y_hat)
        ylogy = MultiplicationNode(learnable=False, is_biased=is_biased, is_hadamard=True)
        input_nodes[1].add(ylogy)

        # Connect logy to the multiplication node
        logy.add(ylogy)

        return ylogy