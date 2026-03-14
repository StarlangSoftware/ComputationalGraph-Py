from abc import ABC, abstractmethod
from typing import List, Set
from Math.Tensor import Tensor
from ComputationalGraph.Node.ComputationalNode import ComputationalNode


class Optimizer(ABC):
    def __init__(self, learning_rate: float, eta_decrease: float):
        """
        Initializes the Optimizer.

        :param learning_rate: The step size for updates.
        :param eta_decrease: The factor by which learning rate is multiplied over time.
        """
        self._learning_rate: float = learning_rate
        self.__eta_decrease: float = eta_decrease

    def setLearningRate(self):
        """Updates the learning rate of the optimizer."""
        self._learning_rate *= self.__eta_decrease

    def __broadcast(self, node: "ComputationalNode") -> int:
        """
        Checks if broadcasting should be applied to the corresponding node.
        Returns the index of the dimension to collapse, or -1 if none.
        """
        v_shape = node.getValue().getShape()
        b_shape = node.getBackward().getShape()

        # Ensure ranks match; if they don't, broadcasting logic might need more complexity
        if len(v_shape) != len(b_shape):
            pass

        index = -1
        for i in range(len(v_shape)):
            if v_shape[i] != b_shape[i]:
                if v_shape[i] == 1:
                    if index != -1:
                        return -1
                    index = i
                else:
                    raise ValueError("Value and Backward shapes are not compatible")
        return index

    def __updateRecursive(self, visited: Set["ComputationalNode"], node: "ComputationalNode"):
        """Recursive helper function to update the values of learnable nodes."""
        visited.add(node)

        if node.isLearnable() and node.getBackward() is not None:
            index = self.__broadcast(node)

            # Handling the case where backward tensor needs to be summed to match value shape
            if index != -1:
                v_shape = node.getValue().getShape()
                b_shape = node.getBackward().getShape()

                v_prod = 1
                b_prod = 1
                for i in range(len(v_shape) - 1, index - 1, -1):
                    v_prod *= v_shape[i]
                    b_prod *= b_shape[i]

                backward_data = node.getBackward().getData()
                # Initialize result list with zeros matching value total size
                result_values = [0.0] * len(node.getValue().getData())

                i = 0
                while i < len(backward_data):
                    for j in range(i, i + b_prod):
                        # Logic: values[((j - i) % v) + v * (j / b)] += backwardValues.get(j);
                        target_idx = ((j - i) % v_prod) + v_prod * (j // b_prod)
                        result_values[target_idx] += backward_data[j]
                    i += b_prod

                node.setBackward(Tensor(result_values, v_shape))

            # Apply specific optimizer logic (SGD, Adam, etc.)
            self.setGradients(node)
            node.updateValue()

        # Recurse through children
        for t in range(node.childrenSize()):
            child = node.getChild(t)
            if child not in visited:
                self.__updateRecursive(visited, child)

    @abstractmethod
    def setGradients(self, node: "ComputationalNode"):
        """Sets the gradients (backward values) of the node based on learning rate."""
        pass

    def updateValues(self, leaf_nodes: List["ComputationalNode"]):
        """Updates the values of all learnable nodes in the graph."""
        visited: Set["ComputationalNode"] = set()
        for node in leaf_nodes:
            if node not in visited:
                self.__updateRecursive(visited, node)