from typing import Dict
from ComputationalGraph.Node.ComputationalNode import ComputationalNode

class ConcatenatedNode(ComputationalNode):
    def __init__(self, dimension: int):
        """
        Initializes a ConcatenatedNode.

        :param dimension: The axis along which the parent tensors will be concatenated.
        """
        super().__init__(learnable=False, is_biased=False)
        self.__index_map: Dict[ComputationalNode, int] = {}
        self.__dimension: int = dimension

    def getDimension(self) -> int:
        """Returns the dimension/axis of concatenation."""
        return self.__dimension

    def getIndex(self, node: ComputationalNode) -> int:
        """Returns the order/index of a specific parent node in the concatenation."""
        return self.__index_map.get(node)

    def addNode(self, node: ComputationalNode):
        """
        Registers a node into the index map.
        """
        self.__index_map[node] = len(self.__index_map)