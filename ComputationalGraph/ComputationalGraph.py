from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional
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

    def __init__(self, parameters: NeuralNetworkParameter):
        self.input_nodes: list[ComputationalNode] = []
        self.output_node: Optional[ComputationalNode] = None
        self.parameters = parameters
        self._leaf_nodes: Optional[list[ComputationalNode]] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, train_set: list[Tensor]):
        pass

    @abstractmethod
    def test(self, test_set: list[Tensor]):
        pass

    @abstractmethod
    def getOutputValue(self, output_node: ComputationalNode) -> list[float]:
        pass

    # ------------------------------------------------------------------
    # Edge construction helpers
    # ------------------------------------------------------------------

    def addEdge(self, first: ComputationalNode, second, is_biased: bool = False) -> ComputationalNode:
        if isinstance(second, Function):
            return second.addEdge([first], is_biased)
        elif isinstance(second, MultiplicationNode):
            new_node = MultiplicationNode(
                learnable=False,
                is_biased=is_biased,
                is_hadamard=second.isHadamard(),
                priority_node=first
            )
            first.addChild(new_node)
            new_node.addParent(first)
            second.addChild(new_node)
            new_node.addParent(second)
            return new_node
        else:
            raise ValueError("Illegal type for argument 'second'")

    def addFunctionEdge(self, input_nodes: list[ComputationalNode], second: Function, is_biased: bool = False) -> ComputationalNode:
        return second.addEdge(input_nodes, is_biased)

    def addEdgeHadamard(self, first: ComputationalNode, second: ComputationalNode,
                        is_biased: bool, is_hadamard: bool) -> ComputationalNode:
        new_node = MultiplicationNode(learnable=False, is_biased=is_biased, is_hadamard=is_hadamard, priority_node=first)
        first.addChild(new_node)
        new_node.addParent(first)
        second.addChild(new_node)
        new_node.addParent(second)
        return new_node

    def addAdditionEdge(self, first: ComputationalNode, second: ComputationalNode,
                        is_biased: bool = False) -> ComputationalNode:
        new_node = ComputationalNode(learnable=False, is_biased=is_biased)
        first.addChild(new_node)
        new_node.addParent(first)
        second.addChild(new_node)
        new_node.addParent(second)
        return new_node

    def concatEdges(self, nodes: list[ComputationalNode], dimension: int) -> ComputationalNode:
        new_node = ConcatenatedNode(dimension)
        for node in nodes:
            node.addChild(new_node)
            new_node.addParent(node)
            new_node.addNode(node)
        return new_node

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def _sortRecursive(self, node: ComputationalNode, visited: set) -> deque:
        queue = deque()
        visited.add(node)
        for i in range(node.childrenSize()):
            child = node.getChild(i)
            if child not in visited:
                queue.extend(self._sortRecursive(child, visited))
        queue.append(node)
        return queue

    def _topologicalSort(self) -> deque:
        sorted_list = deque()
        visited = set()
        for node in self._leaf_nodes:
            if node not in visited:
                q = self._sortRecursive(node, visited)
                sorted_list.extend(q)
        return sorted_list

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def _clearRecursive(self, visited: set, node: ComputationalNode):
        visited.add(node)
        if not node.isLearnable():
            node.setValue(None)
        node.setBackward(None)
        for i in range(node.childrenSize()):
            child = node.getChild(i)
            if child not in visited:
                self._clearRecursive(visited, child)

    def _clear(self):
        visited = set()
        for node in self._leaf_nodes:
            if node not in visited:
                self._clearRecursive(visited, node)

    # ------------------------------------------------------------------
    # Bias helpers
    # ------------------------------------------------------------------

    def _transposeAxes(self, length: int) -> tuple[int, ...]:
        axes = list(range(length))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        return tuple(axes)

    def _getBiasedPartial(self, tensor: Tensor) -> Tensor:
        shape = tensor.getShape()
        end_indices = list(shape)
        end_indices[-1] -= 1

        start_indices = (0,) * len(shape)

        return tensor.partial(start_indices, tuple(end_indices))

    def _getBiased(self, node: ComputationalNode):
        shape = node.getValue().getShape()
        last_dim = shape[-1]
        old_values = list(node.getValue().getData())
        values = []
        for i, v in enumerate(old_values):
            values.append(v)
            if (i + 1) % last_dim == 0:
                values.append(1.0)
        new_shape = list(shape)
        new_shape[-1] += 1
        node.setValue(Tensor(values, tuple(new_shape)))

    # ------------------------------------------------------------------
    # Derivative calculation
    # ------------------------------------------------------------------

    def _calculateDerivative(self, node: ComputationalNode, child: ComputationalNode) -> Optional[Tensor]:
        if child.parentsSize() == 0:
            return None

        backward = self._getBiasedPartial(child.getBackward()) if child.isBiased() else child.getBackward()

        if isinstance(child, FunctionNode):
            function = child.getFunction()
            child_value = self._getBiasedPartial(child.getValue()) if child.isBiased() else child.getValue()
            return function.derivative(child_value, backward)

        if isinstance(child, ConcatenatedNode):
            index = child.getIndex(node)
            block_size = backward.getShape()[child.getDimension()] // child.parentsSize()
            dimensions = block_size
            shape = list(backward.getShape())
            for i in range(len(shape)):
                dim = child.getDimension()
                if dim > i:
                    pass
                elif dim < i:
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
            if left is node:
                right_value = right.getValue()
                if child.isHadamard():
                    return right_value.hadamardProduct(backward)
                return backward.multiply(right_value.transpose(self._transposeAxes(len(right_value.getShape()))))
            left_value = left.getValue()
            if child.isHadamard():
                return left_value.hadamardProduct(backward)
            if left_value is not None and backward is not None:
                return left_value.transpose(self._transposeAxes(len(left_value.getShape()))).multiply(backward)
            raise ValueError("Backward and/or left child values are None")

        # Plain addition node
        return backward

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def backpropagation(self):
        sorted_nodes = self._topologicalSort()
        if not sorted_nodes:
            return

        output_node = sorted_nodes.popleft()
        batch_size = self.parameters.getBatchSize()
        backward = [1.0 / batch_size] * len(list(output_node.getValue().getData()))
        output_node.setBackward(Tensor(backward, output_node.getValue().getShape()))

        while sorted_nodes:
            node = sorted_nodes.popleft()
            if node.childrenSize() > 0:
                for i in range(node.childrenSize()):
                    child = node.getChild(i)
                    derivative = self._calculateDerivative(node, child)
                    if derivative is not None:
                        if node.getBackward() is None:
                            node.setBackward(derivative)
                        else:
                            node.setBackward(node.getBackward().add(derivative))

        self.parameters.getOptimizer().updateValues(self._leaf_nodes)
        self._clear()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _findOutputNode(self, node: ComputationalNode) -> ComputationalNode:
        if node.childrenSize() == 0:
            return node
        return self._findOutputNode(node.getChild(0))

    def _findLeafNodes(self) -> list[ComputationalNode]:
        leaf_nodes = []
        output_node = self._findOutputNode(self.input_nodes[0])
        queue = [output_node]
        visited = set()
        while queue:
            current = queue.pop(0)
            if current.parentsSize() == 0:
                leaf_nodes.append(current)
            for i in range(current.parentsSize()):
                parent = current.getParent(i)
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)
        return leaf_nodes

    def predict(self) -> list[float]:
        class_labels = self._forwardCalculation(enable_dropout=False)
        self._clear()
        return class_labels

    def forwardCalculation(self) -> list[float]:
        if self._leaf_nodes is None:
            self._leaf_nodes = self._findLeafNodes()
        return self._forwardCalculation(enable_dropout=True)

    def _forwardCalculation(self, enable_dropout: bool) -> list[float]:
        sorted_nodes = self._topologicalSort()
        if not sorted_nodes:
            return []

        concatenated_node_map: dict[ConcatenatedNode, list[Optional[ComputationalNode]]] = {}
        counter_map: dict[ComputationalNode, int] = {}

        while len(sorted_nodes) > 1:
            current_node = sorted_nodes.pop()  # removeLast

            if current_node.isBiased():
                self._getBiased(current_node)

            if current_node.getValue() is None:
                raise ValueError("Current node's value is None")

            if current_node.childrenSize() > 0:
                if current_node is self.output_node and not enable_dropout:
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
                                if any(n is None for n in nodes_arr):
                                    raise ValueError("Concatenation nodes missing parents")
                                nodes_arr = [n for n in nodes_arr if n is not None]
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
                            elif child.getPriorityNode() is not current_node:
                                child.setValue(child_value.multiply(current_value))
                            else:
                                child.setValue(current_value.multiply(child_value))
                        else:
                            child.setValue(child.getValue().add(current_node.getValue()))

        return self.getOutputValue(self.output_node)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, file_name: str):
        try:
            with open(file_name, 'wb') as f:
                pickle.dump(self, f)
        except IOError:
            print("Object could not be saved.")

    @staticmethod
    def loadModel(file_name: str) -> Optional[ComputationalGraph]:
        try:
            with open(file_name, 'rb') as f:
                return pickle.load(f)
        except (IOError, pickle.UnpicklingError):
            return None
