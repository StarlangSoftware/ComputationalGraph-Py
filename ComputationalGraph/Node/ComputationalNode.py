from typing import List, Optional

from Math.Tensor import Tensor


class ComputationalNode:
    """
    Represents a node in the computational graph.
    """

    _value: Optional[Tensor]
    _backward: Optional[Tensor]
    _is_biased: bool
    _learnable: bool
    __children: List['ComputationalNode']
    __parents: List['ComputationalNode']

    def __init__(
        self,
        learnable: bool = False,
        is_biased: bool = False,
        value: Optional[Tensor] = None
    ) -> None:
        """
        Initializes a ComputationalNode.

        :param learnable: Indicates if the node parameters can be updated.
        :param is_biased: Indicates whether the node is biased.
        :param value: The tensor value associated with the node.
        """
        self._value = value
        self._backward = None
        self._is_biased = is_biased
        self._learnable = learnable
        self.__children = []
        self.__parents = []

    def getChild(self, index: int) -> 'ComputationalNode':
        """
        Returns the child node at the given index.

        :param index: Child index.
        :return: Child computational node.
        """
        return self.__children[index]

    def addChild(self, child: 'ComputationalNode') -> None:
        """
        Adds a child node.

        :param child: Child computational node.
        """
        self.__children.append(child)

    def addParent(self, parent: 'ComputationalNode') -> None:
        """
        Adds a parent node.

        :param parent: Parent computational node.
        """
        self.__parents.append(parent)

    def add(self, child: 'ComputationalNode') -> None:
        """
        Adds a child node and sets this node as its parent.

        :param child: Child computational node.
        """
        self.__children.append(child)
        child.addParent(self)

    def getParent(self, index: int) -> 'ComputationalNode':
        """
        Returns the parent node at the given index.

        :param index: Parent index.
        :return: Parent computational node.
        """
        return self.__parents[index]

    def childrenSize(self) -> int:
        """
        Returns the number of child nodes.

        :return: Number of child nodes.
        """
        return len(self.__children)

    def parentsSize(self) -> int:
        """
        Returns the number of parent nodes.

        :return: Number of parent nodes.
        """
        return len(self.__parents)

    def isLearnable(self) -> bool:
        """
        Returns whether the node is learnable.

        :return: True if the node is learnable, False otherwise.
        """
        return self._learnable

    def isBiased(self) -> bool:
        """
        Returns whether the node is biased.

        :return: True if the node is biased, False otherwise.
        """
        return self._is_biased

    def getValue(self) -> Optional[Tensor]:
        """
        Returns the value tensor of the node.

        :return: Value tensor.
        """
        return self._value

    def setValue(self, value: Optional[Tensor]) -> None:
        """
        Sets the value tensor of the node.

        :param value: New value tensor.
        """
        self._value = value

    def getBackward(self) -> Optional[Tensor]:
        """
        Returns the backward tensor of the node.

        :return: Backward tensor.
        """
        return self._backward

    def setBackward(self, backward: Optional[Tensor]) -> None:
        """
        Sets the backward tensor of the node.

        :param backward: New backward tensor.
        """
        self._backward = backward

    def updateValue(self) -> None:
        """
        Updates the value by adding the backward tensor.

        :raises ValueError: If value or backward tensor is None.
        """
        if self._value is not None and self._backward is not None:
            self.setValue(self._value.add(self._backward))
        else:
            raise ValueError("Cannot update value: value or backward tensor is None.")

    def toString(self) -> str:
        """
        Returns string representation of the node.

        :return: String representation.
        """
        details = []
        if self._value is not None:
            shape_str = ", ".join(map(str, self._value.getShape()))
            details.append(f"Value Shape: [{shape_str}]")

        details.append(f"is learnable: {self._learnable}")
        details.append(f"is biased: {self._is_biased}")

        return f"Node({', '.join(details)})"

    def __repr__(self) -> str:
        """
        Returns string representation of the node.

        :return: String representation.
        """
        return self.toString()