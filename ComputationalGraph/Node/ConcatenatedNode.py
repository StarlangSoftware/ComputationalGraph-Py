from typing import Dict, Optional

from ComputationalGraph.Node.ComputationalNode import ComputationalNode


class ConcatenatedNode(ComputationalNode):
    """
    Represents a node that concatenates its parent nodes along a given dimension.
    """

    __index_map: Dict[ComputationalNode, int]
    __dimension: int

    def __init__(self, dimension: int) -> None:
        """
        Initializes a ConcatenatedNode.

        :param dimension: The axis along which the parent tensors will be concatenated.
        """
        super().__init__(learnable=False, is_biased=False)
        self.__index_map = {}
        self.__dimension = dimension

    def getDimension(self) -> int:
        """
        Returns the dimension of concatenation.

        :return: Concatenation dimension.
        """
        return self.__dimension

    def getIndex(self, node: ComputationalNode) -> Optional[int]:
        """
        Returns the index of the given node in the concatenation order.

        :param node: Input computational node.
        :return: Index of the node.
        """
        return self.__index_map.get(node)

    def addNode(self, node: ComputationalNode) -> None:
        """
        Registers a node into the index map.

        :param node: Input computational node.
        """
        self.__index_map[node] = len(self.__index_map)