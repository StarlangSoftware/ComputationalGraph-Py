from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Function.Function import Function


class FunctionNode(ComputationalNode):
    """
    Node representing a function in the computational graph.
    """

    __function: Function

    def constructor1(self, is_biased: bool, function: Function) -> None:
        """
        Constructor with bias flag and function.

        :param is_biased: Bias flag.
        :param function: Function object.
        """
        super().__init__(False, is_biased)
        self.__function = function

    def constructor2(self, learnable: bool, is_biased: bool, function: Function) -> None:
        """
        Constructor with learnable flag, bias flag, and function.

        :param learnable: Learnable flag.
        :param is_biased: Bias flag.
        :param function: Function object.
        """
        super().__init__(learnable, is_biased)
        self.__function = function

    def __init__(self, *args, **kwargs):
        """
        Main constructor supporting both positional and keyword usage.
        """
        if kwargs:
            learnable = kwargs.get("learnable", False)
            is_biased = kwargs.get("is_biased", False)
            function = kwargs.get("function", None)

            if function is None:
                raise ValueError("FunctionNode requires a function argument")

            self.constructor2(learnable, is_biased, function)
            return

        if len(args) == 2:
            self.constructor1(args[0], args[1])
        elif len(args) == 3:
            self.constructor2(args[0], args[1], args[2])
        else:
            raise ValueError("Invalid constructor arguments for FunctionNode")

    def getFunction(self) -> Function:
        """
        Getter for function.

        :return: Function object.
        """
        return self.__function

    def toString(self) -> str:
        """
        Returns string representation.

        :return: String representation.
        """
        details = ""

        if self.__function is not None:
            details += "Function: " + str(self.__function)

        if self.getValue() is not None:
            if len(details) > 0:
                details += ", "
            shape = self.getValue().getShape()
            details += "Value Shape: [" + str(shape[0])
            for i in range(1, len(shape)):
                details += ", " + str(shape[i])
            details += "]"

        details += ", is biased: " + str(self.isBiased()).lower()

        return "FunctionNode(" + details + ")"

    def __str__(self) -> str:
        """
        Returns string representation.

        :return: String representation.
        """
        return self.toString()