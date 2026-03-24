from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional, List, Dict
import pickle

from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from ComputationalGraph.Node.ConcatenatedNode import ConcatenatedNode
from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Function.Dropout import Dropout
from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from Math.Tensor import Tensor


class ComputationalGraph(ABC):
    """
    Abstract computational graph class.
    """

    input_nodes: List[ComputationalNode]
    output_node: Optional[ComputationalNode]
    parameters: NeuralNetworkParameter
    __leaf_nodes: Optional[List[ComputationalNode]]

    def __init__(self, parameters: NeuralNetworkParameter):
        """
        Constructor for ComputationalGraph.

        :param parameters: Neural network parameters.
        """
        self.input_nodes = []
        self.output_node = None
        self.parameters = parameters
        self.__leaf_nodes = None

    @abstractmethod
    def train(self, train_set: List[Tensor]) -> None:
        """
        Trains the computational graph.

        :param train_set: Training set.
        """
        pass

    @abstractmethod
    def test(self, test_set: List[Tensor]):
        """
        Tests the computational graph.

        :param test_set: Test set.
        """
        pass

    @abstractmethod
    def getOutputValue(self, output_node: ComputationalNode) -> List[float]:
        """
        Returns output value of the graph.

        :param output_node: Output node.
        :return: Output values.
        """
        pass

    def addEdge(self,
                first: ComputationalNode,
                second,
                is_biased: bool = False,
                is_hadamard: Optional[bool] = None) -> ComputationalNode:
        """
        Adds an edge to the graph.

        :param first: First computational node.
        :param second: Function, MultiplicationNode, or ComputationalNode.
        :param is_biased: Bias flag.
        :param is_hadamard: Hadamard flag for node-node multiplication.
        :return: New computational node.
        """
        if is_hadamard is not None:
            if not isinstance(second, ComputationalNode):
                raise ValueError("Illegal type for argument 'second'")
            new_node = MultiplicationNode(False, is_biased, is_hadamard, first)
            first.addChild(new_node)
            new_node.addParent(first)
            second.addChild(new_node)
            new_node.addParent(second)
            return new_node

        if isinstance(second, Function):
            return second.addEdge([first], is_biased)

        if isinstance(second, MultiplicationNode):
            new_node = MultiplicationNode(False, is_biased, second.isHadamard(), first)
            first.addChild(new_node)
            new_node.addParent(first)
            second.addChild(new_node)
            new_node.addParent(second)
            return new_node

        raise ValueError("Illegal type for argument 'second'")

    def addFunctionEdge(self,
                        input_nodes: List[ComputationalNode],
                        second: Function,
                        is_biased: bool) -> ComputationalNode:
        """
        Adds a function edge.

        :param input_nodes: Input nodes.
        :param second: Function object.
        :param is_biased: Bias flag.
        :return: New computational node.
        """
        return second.addEdge(input_nodes, is_biased)

    def addAdditionEdge(self,
                        first: ComputationalNode,
                        second: ComputationalNode,
                        is_biased: bool) -> ComputationalNode:
        """
        Adds an addition edge.

        :param first: First node.
        :param second: Second node.
        :param is_biased: Bias flag.
        :return: New computational node.
        """
        new_node = ComputationalNode(False, is_biased)
        first.addChild(new_node)
        new_node.addParent(first)
        second.addChild(new_node)
        new_node.addParent(second)
        return new_node

    def concatEdges(self,
                    nodes: List[ComputationalNode],
                    dimension: int) -> ComputationalNode:
        """
        Concatenates nodes along a dimension.

        :param nodes: Nodes to concatenate.
        :param dimension: Concatenation dimension.
        :return: Concatenated node.
        """
        new_node = ConcatenatedNode(dimension)
        for node in nodes:
            node.addChild(new_node)
            new_node.addParent(node)
            new_node.addNode(node)
        return new_node

    def __sortRecursive(self,
                        node: ComputationalNode,
                        visited: set) -> deque:
        """
        Recursive helper for topological sorting.

        :param node: Current node.
        :param visited: Visited node set.
        :return: Queue of nodes.
        """
        queue = deque()
        visited.add(node)

        for i in range(node.childrenSize()):
            child = node.getChild(i)
            if child not in visited:
                queue.extend(self.__sortRecursive(child, visited))

        queue.append(node)
        return queue

    def __topologicalSort(self) -> deque:
        """
        Performs topological sorting.

        :return: Sorted node list.
        """
        sorted_list = deque()
        visited = set()

        for node in self.__leaf_nodes:
            if node not in visited:
                queue = self.__sortRecursive(node, visited)
                while queue:
                    sorted_list.append(queue.popleft())

        return sorted_list

    def __clearRecursive(self, visited: set, node: ComputationalNode) -> None:
        """
        Recursive helper for clearing graph values.

        :param visited: Visited nodes.
        :param node: Current node.
        """
        visited.add(node)

        if not node.isLearnable():
            node.setValue(None)

        node.setBackward(None)

        for i in range(node.childrenSize()):
            child = node.getChild(i)
            if child not in visited:
                self.__clearRecursive(visited, child)

    def __clear(self) -> None:
        """
        Clears graph node values and gradients.
        """
        visited = set()

        for node in self.__leaf_nodes:
            if node not in visited:
                self.__clearRecursive(visited, node)

    def __transposeAxes(self, length: int) -> tuple[int, ...]:
        """
        Swaps the last two axes.

        :param length: Number of dimensions.
        :return: Transposed axes.
        """
        axes = list(range(length))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        return tuple(axes)

    def __getBiasedPartial(self, tensor: Tensor) -> Tensor:
        """
        Removes bias term from tensor.

        :param tensor: Input tensor.
        :return: Tensor without bias.
        """
        shape = tensor.getShape()
        end_indexes = list(shape)
        end_indexes[-1] -= 1
        start_indexes = (0,) * len(shape)

        return tensor.partial(start_indexes, tuple(end_indexes))

    def __getBiased(self, tensor: ComputationalNode) -> None:
        """
        Appends bias term to tensor value.

        :param tensor: Computational node.
        """
        last_dimension_size = tensor.getValue().getShape()[-1]
        values = []
        old_values = list(tensor.getValue().getData())

        for i, value in enumerate(old_values):
            values.append(value)
            if (i + 1) % last_dimension_size == 0:
                values.append(1.0)

        shape = list(tensor.getValue().getShape())
        shape[-1] += 1

        biased_value = Tensor(values, tuple(shape))
        tensor.setValue(biased_value)

    def __calculateDerivative(self,
                              node: ComputationalNode,
                              child: ComputationalNode) -> Optional[Tensor]:
        """
        Calculates derivative from child to parent.

        :param node: Parent node.
        :param child: Child node.
        :return: Derivative tensor.
        """
        if child.parentsSize() == 0:
            return None

        if child.isBiased():
            backward = self.__getBiasedPartial(child.getBackward())
        else:
            backward = child.getBackward()

        if isinstance(child, FunctionNode):
            function = child.getFunction()

            if child.isBiased():
                child_value = self.__getBiasedPartial(child.getValue())
            else:
                child_value = child.getValue()

            return function.derivative(child_value, backward)

        if isinstance(child, ConcatenatedNode):
            index = child.getIndex(node)
            block_size = backward.getShape()[child.getDimension()] // child.parentsSize()

            dimensions = block_size
            shape = list(backward.getShape())

            for i in range(len(shape)):
                if child.getDimension() > i:
                    pass
                elif child.getDimension() < i:
                    dimensions *= shape[i]
                else:
                    shape[i] = block_size

            child_values = list(backward.getData())
            new_values = []

            i = index * dimensions
            while i < len(child_values):
                for k in range(dimensions):
                    new_values.append(child_values[i + k])
                i += child.parentsSize() * dimensions

            return Tensor(new_values, tuple(shape))

        if isinstance(child, MultiplicationNode):
            left = child.getParent(0)
            right = child.getParent(1)

            if left == node:
                right_value = right.getValue()
                if child.isHadamard():
                    return right_value.hadamardProduct(backward)

                return backward.multiply(
                    right_value.transpose(self.__transposeAxes(len(right_value.getShape())))
                )

            left_value = left.getValue()

            if child.isHadamard():
                return left_value.hadamardProduct(backward)

            if left_value is not None and backward is not None:
                return left_value.transpose(
                    self.__transposeAxes(len(left_value.getShape()))
                ).multiply(backward)

            raise ValueError("Backward and/or left child values are None")

        return backward

    def backpropagation(self) -> None:
        """
        Performs backpropagation.
        """
        sorted_nodes = self.__topologicalSort()
        if not sorted_nodes:
            return

        output_node = sorted_nodes.popleft()

        backward = [1.0 / self.parameters.getBatchSize()] * len(output_node.getValue().getData())
        output_node.setBackward(Tensor(backward, output_node.getValue().getShape()))

        while sorted_nodes:
            node = sorted_nodes.popleft()

            if node.childrenSize() > 0:
                for i in range(node.childrenSize()):
                    child = node.getChild(i)
                    derivative = self.__calculateDerivative(node, child)

                    if derivative is not None:
                        if node.getBackward() is None:
                            node.setBackward(derivative)
                        else:
                            node.setBackward(node.getBackward().add(derivative))

        self.parameters.getOptimizer().updateValues(self.__leaf_nodes)
        self.__clear()

    def predict(self) -> List[float]:
        """
        Performs prediction.

        :return: Predicted class labels.
        """
        class_labels = self.__forwardCalculation(False)
        self.__clear()
        return class_labels

    def forwardCalculation(self) -> List[float]:
        """
        Performs forward calculation for training.

        :return: Predicted class labels.
        """
        if self.__leaf_nodes is None:
            self.__leaf_nodes = self.__findLeafNodes()

        return self.__forwardCalculation(True)

    def __findOutputNode(self, node: ComputationalNode) -> ComputationalNode:
        """
        Finds output node recursively.

        :param node: Starting node.
        :return: Output node.
        """
        if node.childrenSize() == 0:
            return node

        return self.__findOutputNode(node.getChild(0))

    def __findLeafNodes(self) -> List[ComputationalNode]:
        """
        Finds leaf nodes of graph.

        :return: Leaf nodes.
        """
        leaf_nodes = []
        output_node = self.__findOutputNode(self.input_nodes[0])

        queue = [output_node]
        visited = set()

        while queue:
            current_node = queue.pop(0)

            if current_node.parentsSize() == 0:
                leaf_nodes.append(current_node)

            for i in range(current_node.parentsSize()):
                parent = current_node.getParent(i)
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)

        return leaf_nodes

    def __forwardCalculation(self, enable_dropout: bool) -> List[float]:
        """
        Performs forward pass.

        :param enable_dropout: Whether dropout is enabled.
        :return: Output values.
        """
        sorted_nodes = self.__topologicalSort()
        if not sorted_nodes:
            return []

        concatenated_node_map: Dict[ConcatenatedNode, List[Optional[ComputationalNode]]] = {}
        counter_map: Dict[ComputationalNode, int] = {}

        while len(sorted_nodes) > 1:
            current_node = sorted_nodes.pop()

            if current_node.isBiased():
                self.__getBiased(current_node)

            if current_node.getValue() is None:
                print("DEBUG NODE TYPE:", type(current_node))
                print("DEBUG NODE:", current_node)
                print("DEBUG PARENTS:", current_node.parentsSize())
                print("DEBUG CHILDREN:", current_node.childrenSize())
                raise ValueError("Current node's value is None")

            if current_node.childrenSize() > 0:
                if current_node == self.output_node and not enable_dropout:
                    break

                for t in range(current_node.childrenSize()):
                    child = current_node.getChild(t)

                    if child.getValue() is None:
                        if isinstance(child, FunctionNode):
                            function = child.getFunction()
                            current_value = current_node.getValue()

                            if isinstance(function, Dropout):
                                if enable_dropout:
                                    child.setValue(function.calculate(current_value))
                                else:
                                    child.setValue(Tensor(current_value.getData(), current_value.getShape()))
                            else:
                                child.setValue(function.calculate(current_value))

                        elif isinstance(child, ConcatenatedNode):
                            if child not in concatenated_node_map:
                                concatenated_node_map[child] = [None] * child.parentsSize()

                            concatenated_node_map[child][child.getIndex(current_node)] = current_node
                            counter_map[child] = counter_map.get(child, 0) + 1

                            if child.parentsSize() == counter_map[child]:
                                nodes_arr = concatenated_node_map[child]
                                nodes_arr = [node for node in nodes_arr if node is not None]

                                child.setValue(nodes_arr[0].getValue())
                                for i in range(1, len(nodes_arr)):
                                    child.setValue(
                                        child.getValue().concat(nodes_arr[i].getValue(), child.getDimension())
                                    )
                        else:
                            child.setValue(current_node.getValue())

                    else:
                        if isinstance(child, MultiplicationNode):
                            child_value = child.getValue()
                            current_value = current_node.getValue()

                            if child.isHadamard():
                                child.setValue(child_value.hadamardProduct(current_value))
                            elif child.getPriorityNode() != current_node:
                                child.setValue(child_value.multiply(current_value))
                            else:
                                child.setValue(current_value.multiply(child_value))
                        else:
                            result = child.getValue()
                            current_value = current_node.getValue()
                            child.setValue(result.add(current_value))

        return self.getOutputValue(self.output_node)

    def save(self, file_name: str) -> None:
        """
        Saves model.

        :param file_name: File name.
        """
        try:
            with open(file_name, "wb") as output_file:
                pickle.dump(self, output_file)
        except OSError:
            print("Object could not be saved.")

    @staticmethod
    def loadModel(file_name: str) -> Optional["ComputationalGraph"]:
        """
        Loads model.

        :param file_name: File name.
        :return: Loaded model.
        """
        try:
            with open(file_name, "rb") as input_file:
                return pickle.load(input_file)
        except (OSError, pickle.UnpicklingError):
            return None