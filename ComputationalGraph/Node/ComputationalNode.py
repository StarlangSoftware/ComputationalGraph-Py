from typing import List, Optional, TYPE_CHECKING
from Math.Tensor import Tensor

class ComputationalNode:
    def __init__(
        self,
        learnable: bool = False,
        is_biased: bool = False,
        value: Optional[Tensor] = None
    ):
        """
        Initializes a ComputationalNode.

        :param learnable: Indicates if the node parameters can be updated.
        :param is_biased: Indicates whether the node is biased.
        :param value: The tensor value associated with the node.
        """
        self._value: Optional[Tensor] = value
        self._backward: Optional[Tensor] = None
        self._is_biased: bool = is_biased
        self._learnable: bool = learnable
        self.__children: List[ComputationalNode] = []
        self.__parents: List[ComputationalNode] = []

    # --- Parent/Child Management ---

    def getChild(self, index: int) -> 'ComputationalNode':
        return self.__children[index]

    def addChild(self, child: 'ComputationalNode'):
        self.__children.append(child)

    def addParent(self, parent: 'ComputationalNode'):
        self.__parents.append(parent)

    def add(self, child: 'ComputationalNode'):
        """Adds a child and automatically sets this node as the child's parent."""
        self.__children.append(child)
        child.addParent(self)

    def getParent(self, index: int) -> 'ComputationalNode':
        return self.__parents[index]

    def childrenSize(self) -> int:
        return len(self.__children)

    def parentsSize(self) -> int:
        return len(self.__parents)

    # --- Getters and Setters ---

    def isLearnable(self) -> bool:
        return self._learnable

    def isBiased(self) -> bool:
        return self._is_biased

    def getValue(self) -> Optional[Tensor]:
        return self._value

    def setValue(self, value: Optional[Tensor]) -> None:
        self._value = value

    def getBackward(self) -> Optional[Tensor]:
        return self._backward

    def setBackward(self, backward: Optional[Tensor]) -> None:
        self._backward = backward

    # --- Logic ---

    def updateValue(self):
        """Updates the value by adding the backward (gradient) tensor."""
        if self._value is not None and self._backward is not None:
            self.setValue(self._value.add(self._backward))
        else:
            raise ValueError("Cannot update value: value or backward tensor is None.")

    def __repr__(self) -> str:
        """String representation of the node, mimicking the Java toString."""
        details = []
        if self._value is not None:
            shape_str = ", ".join(map(str, self._value.getShape()))
            details.append(f"Value Shape: [{shape_str}]")

        details.append(f"is learnable: {self._learnable}")
        details.append(f"is biased: {self._is_biased}")

        return f"Node({', '.join(details)})"