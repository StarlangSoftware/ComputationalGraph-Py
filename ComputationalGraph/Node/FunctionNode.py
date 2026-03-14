from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Function.Function import Function

class FunctionNode(ComputationalNode):
    def __init__(
        self,
        function: Function,
        is_biased: bool = False,
        learnable: bool = False
    ):
        """
        Initializes a FunctionNode.

        :param function: The mathematical function instance (e.g., Sigmoid, ReLU).
        :param is_biased: Indicates whether the node is biased.
        :param learnable: Indicates if the node has learnable parameters.
        """
        super().__init__(learnable=learnable, is_biased=is_biased)
        self.__function = function

    def getFunction(self) -> Function:
        return self.__function

    def __repr__(self) -> str:
        """String representation mirroring the Java toString logic."""
        details = []

        if self.__function:
            details.append(f"Function: {self.__function}")

        if self._value is not None:
            shape_str = ", ".join(map(str, self._value.getShape()))
            details.append(f"Value Shape: [{shape_str}]")

        details.append(f"is biased: {self._is_biased}")

        return f"FunctionNode({', '.join(details)})"