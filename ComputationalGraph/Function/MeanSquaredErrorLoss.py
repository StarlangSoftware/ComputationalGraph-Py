from typing import List

from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from .Negation import Negation
from .Power import Power

class MeanSquaredErrorLoss(Negation):
    """
    Mean Squared Error (MSE) Loss function representation for the computational graph.
    Inherits from Negation to compute the -y portion of the (y_hat - y)^2 operation.
    """

    def addEdge(self, input_nodes: List[ComputationalNode], is_biased: bool) -> ComputationalNode:
        """
        Constructs the subgraph for the mean squared error loss operation.

        :param input_nodes: A list of input computational nodes (y and y_hat).
        :param is_biased: Indicates whether the connection is biased.
        :return: The final Power node representing the squared difference.
        """
        # Create a negation node for the first input (acting as -y)
        # using 'self' since this class inherits the Negation math
        negated_y = FunctionNode(function=self, is_biased=False)
        input_nodes[0].add(negated_y)

        # Create a standard node to act as an addition/accumulation point for (y_hat - y)
        y_minus_negated_y = ComputationalNode()
        negated_y.add(y_minus_negated_y)
        input_nodes[1].add(y_minus_negated_y)

        # Create a power node to square the resulting difference
        new_node = FunctionNode(function=Power(), is_biased=is_biased)
        y_minus_negated_y.add(new_node)

        return new_node