from typing import List

from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode
from .Sigmoid import Sigmoid

class SiLU(Sigmoid):
    """
    SiLU (Sigmoid Linear Unit) / Swish activation function representation for the computational graph.
    Inherits from Sigmoid to compute x * sigmoid(x).
    """

    def addEdge(self, input_nodes: List[ComputationalNode], is_biased: bool) -> ComputationalNode:
        """
        Constructs the subgraph for the SiLU activation operation.

        :param input_nodes: A list of input computational nodes.
        :param is_biased: Indicates whether the connection is biased.
        :return: The final multiplication node representing x * sigmoid(x).
        """
        sigmoid = FunctionNode(function=self, is_biased=False)
        input_nodes[0].add(sigmoid)

        # Create a Hadamard multiplication node for the element-wise product
        swish = MultiplicationNode(learnable=False, is_biased=is_biased, is_hadamard=True)

        # Connect the output of the sigmoid to the multiplication node
        sigmoid.add(swish)

        # Connect the original input directly to the multiplication node as well
        input_nodes[0].add(swish)

        return swish