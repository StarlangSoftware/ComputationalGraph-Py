from typing import Optional
from Math.Tensor import Tensor
from ComputationalGraph.Node.ComputationalNode import ComputationalNode


class MultiplicationNode(ComputationalNode):
    """
    A node representing a multiplication operation in the computational graph.
    """

    def __init__(self,
                 learnable: bool = True,
                 is_biased: bool = False,
                 is_hadamard: bool = False,
                 priority_node: Optional['ComputationalNode'] = None,
                 value: Optional[Tensor] = None):
        """
        Initializes a MultiplicationNode.
        Consolidates all Java constructor overloads into a single Python constructor.

        :param learnable: Indicates whether the node is learnable.
        :param is_biased: Indicates whether the node is biased.
        :param is_hadamard: Indicates whether the multiplication is an element-wise (Hadamard) product.
        :param priority_node: An optional priority computational node.
        :param value: The tensor value associated with the node.
        """
        super().__init__(learnable=learnable, is_biased=is_biased, value=value)

        self._is_hadamard: bool = is_hadamard
        self._priority_node: Optional['ComputationalNode'] = priority_node

    def isHadamard(self) -> bool:
        return self._is_hadamard

    def getPriorityNode(self) -> Optional['ComputationalNode']:
        return self._priority_node

    def __str__(self) -> str:
        """
        Returns a string representation of the node.
        Overrides the ComputationalNode __str__ to prefix with "MultiplicationNode".
        """
        details = []

        if self.getValue() is not None:
            shape_str = ", ".join(str(dim) for dim in self.getValue().getShape())
            details.append(f"Value Shape: [{shape_str}]")

        details.append(f"is learnable: {str(self._learnable).lower()}")
        details.append(f"is biased: {str(self._is_biased).lower()}")

        return f"MultiplicationNode({', '.join(details)})"