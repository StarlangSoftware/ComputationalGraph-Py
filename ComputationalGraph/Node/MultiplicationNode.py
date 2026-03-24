from __future__ import annotations
from typing import Optional

from Math.Tensor import Tensor
from ComputationalGraph.Node.ComputationalNode import ComputationalNode


class MultiplicationNode(ComputationalNode):
    """
    A node representing a multiplication operation in the computational graph.
    """

    __is_hadamard: bool
    __priority_node: Optional[ComputationalNode]

    def constructor1(self, learnable: bool, is_biased: bool, is_hadamard: bool) -> None:
        """
        Constructor with learnable, bias and hadamard flags.

        :param learnable: Learnable flag.
        :param is_biased: Bias flag.
        :param is_hadamard: Hadamard flag.
        """
        super().__init__(learnable, is_biased)
        self.__is_hadamard = is_hadamard
        self.__priority_node = None

    def constructor2(self,
                     learnable: bool,
                     is_biased: bool,
                     is_hadamard: bool,
                     priority_node: ComputationalNode) -> None:
        """
        Constructor with priority node.

        :param learnable: Learnable flag.
        :param is_biased: Bias flag.
        :param is_hadamard: Hadamard flag.
        :param priority_node: Priority node.
        """
        super().__init__(learnable, is_biased)
        self.__is_hadamard = is_hadamard
        self.__priority_node = priority_node

    def constructor3(self,
                     learnable: bool,
                     is_biased: bool,
                     value: Tensor,
                     is_hadamard: bool) -> None:
        """
        Constructor with tensor value.

        :param learnable: Learnable flag.
        :param is_biased: Bias flag.
        :param value: Tensor value.
        :param is_hadamard: Hadamard flag.
        """
        super().__init__(learnable, is_biased)
        self.setValue(value)
        self.__priority_node = None
        self.__is_hadamard = is_hadamard

    def constructor4(self, learnable: bool, value: Tensor) -> None:
        """
        Constructor with learnable flag and value.

        :param learnable: Learnable flag.
        :param value: Tensor value.
        """
        super().__init__(learnable, False, value)
        self.setValue(value)
        self.__priority_node = None
        self.__is_hadamard = False

    def constructor5(self, value: Tensor) -> None:
        """
        Constructor with only tensor value.

        :param value: Tensor value.
        """
        super().__init__(True, False)
        self.setValue(value)
        self.__priority_node = None
        self.__is_hadamard = False

    def constructor6(self, learnable: bool, is_biased: bool) -> None:
        """
        Constructor with learnable and bias flags.

        :param learnable: Learnable flag.
        :param is_biased: Bias flag.
        """
        super().__init__(learnable, is_biased)
        self.__priority_node = None
        self.__is_hadamard = False

    def __init__(self, *args, **kwargs):
        """
        Main constructor supporting both positional and keyword usage.
        """
        if kwargs:
            learnable = kwargs.get("learnable", True)
            is_biased = kwargs.get("is_biased", False)
            is_hadamard = kwargs.get("is_hadamard", False)
            priority_node = kwargs.get("priority_node", None)
            value = kwargs.get("value", None)

            if value is not None and priority_node is None:
                self.constructor3(learnable, is_biased, value, is_hadamard)
            elif priority_node is not None:
                self.constructor2(learnable, is_biased, is_hadamard, priority_node)
            else:
                self.constructor1(learnable, is_biased, is_hadamard)

            return

        if len(args) == 1 and isinstance(args[0], Tensor):
            self.constructor5(args[0])
        elif len(args) == 2 and isinstance(args[0], bool) and isinstance(args[1], Tensor):
            self.constructor4(args[0], args[1])
        elif len(args) == 4 and isinstance(args[0], bool) and isinstance(args[1], bool) and isinstance(args[2], Tensor) and isinstance(args[3], bool):
            self.constructor3(args[0], args[1], args[2], args[3])
        elif len(args) == 4 and isinstance(args[0], bool) and isinstance(args[1], bool) and isinstance(args[2], bool) and isinstance(args[3], ComputationalNode):
            self.constructor2(args[0], args[1], args[2], args[3])
        elif len(args) == 3 and isinstance(args[0], bool) and isinstance(args[1], bool) and isinstance(args[2], bool):
            self.constructor1(args[0], args[1], args[2])
        elif len(args) == 2 and isinstance(args[0], bool) and isinstance(args[1], bool):
            self.constructor6(args[0], args[1])
        else:
            raise ValueError("Invalid constructor arguments for MultiplicationNode")

    def isHadamard(self) -> bool:
        """
        Getter for hadamard flag.

        :return: Hadamard flag.
        """
        return self.__is_hadamard

    def getPriorityNode(self) -> Optional[ComputationalNode]:
        """
        Getter for priority node.

        :return: Priority node.
        """
        return self.__priority_node

    def toString(self) -> str:
        """
        Returns string representation.

        :return: String representation.
        """
        details = ""

        if self.getValue() is not None:
            shape = self.getValue().getShape()
            details += "Value Shape: [" + str(shape[0])
            for i in range(1, len(shape)):
                details += ", " + str(shape[i])
            details += "]"

        if len(details) > 0:
            details += ", "

        details += "is learnable: " + str(self.isLearnable()).lower()
        details += ", is biased: " + str(self.isBiased()).lower()

        return "MultiplicationNode(" + details + ")"

    def __str__(self) -> str:
        """
        Returns string representation.

        :return: String representation.
        """
        return self.toString()