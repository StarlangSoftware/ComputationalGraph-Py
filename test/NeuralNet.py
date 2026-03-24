from __future__ import annotations

import random as py_random

from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode
from ComputationalGraph.Function.Sigmoid import Sigmoid
from ComputationalGraph.Function.ELU import ELU
from ComputationalGraph.Function.Softmax import Softmax
from ComputationalGraph.Function.Dropout import Dropout
from Math.Tensor import Tensor


class NeuralNet(ComputationalGraph):

    def __init__(self, parameters: NeuralNetworkParameter):
        super().__init__(parameters)

    def _createInputTensor(self, instance: Tensor) -> Tensor:
        data = [instance.getValue((i,)) for i in range(instance.getShape()[0] - 1)]
        return Tensor(data, (1, instance.getShape()[0] - 1))

    def _setClassLabelNode(self, n: int, class_label: int) -> Tensor:
        data = [1.0 if i == class_label else 0.0 for i in range(n)]
        return Tensor(data, (1, n))

    def train(self, train_set: list[Tensor]):
        # Input nodes
        input_node = MultiplicationNode(learnable=False, is_biased=True)
        class_label_node = ComputationalNode()
        self.input_nodes.append(input_node)
        self.input_nodes.append(class_label_node)

        # First layer
        n_input_with_bias = 5
        n_hidden_1 = 4
        rng = py_random.Random(self.parameters.getSeed())
        t1 = Tensor(self.parameters.initializeWeights(n_input_with_bias, n_hidden_1, rng),
                    (n_input_with_bias, n_hidden_1))
        w1 = MultiplicationNode(value=t1)
        a1 = self.addEdge(input_node, w1)
        a1_sigmoid = self.addEdge(a1, Sigmoid())
        a1_sigmoid_dropout = self.addEdge(
            a1_sigmoid,
            Dropout(self.parameters.getDropout(), py_random.Random(self.parameters.getSeed())),
            is_biased=True
        )

        # Second layer
        n_hidden_2 = 20
        t2 = Tensor(self.parameters.initializeWeights(n_hidden_1 + 1, n_hidden_2, py_random.Random(self.parameters.getSeed())),
                    (n_hidden_1 + 1, n_hidden_2))
        w2 = MultiplicationNode(value=t2)
        a2 = self.addEdge(a1_sigmoid_dropout, w2)
        a2_elu = self.addEdge(a2, ELU(3.0))
        a2_elu_dropout = self.addEdge(
            a2_elu,
            Dropout(self.parameters.getDropout(), py_random.Random(self.parameters.getSeed())),
            is_biased=True
        )

        # Output layer
        n_classes = 3
        t3 = Tensor(self.parameters.initializeWeights(n_hidden_2 + 1, n_classes, py_random.Random(self.parameters.getSeed())),
                    (21, n_classes))
        w3 = MultiplicationNode(value=t3)
        a3 = self.addEdge(a2_elu_dropout, w3)
        self.output_node = self.addEdge(a3, Softmax())

        loss_nodes = [self.output_node, class_label_node]
        self.addFunctionEdge(loss_nodes, self.parameters.getLossFunction(), is_biased=False)

        # Training loop
        for epoch in range(self.parameters.getEpoch()):
            rng_shuffle = py_random.Random(self.parameters.getSeed())
            for j in range(len(train_set)):
                i1 = rng_shuffle.randint(0, len(train_set) - 1)
                i2 = rng_shuffle.randint(0, len(train_set) - 1)
                train_set[i1], train_set[i2] = train_set[i2], train_set[i1]

            for instance in train_set:
                input_node.setValue(self._createInputTensor(instance))
                class_label = int(instance.getValue((instance.getShape()[0] - 1,)))
                class_label_node.setValue(self._setClassLabelNode(n_classes, class_label))
                self.forwardCalculation()
                self.backpropagation()

            self.parameters.getOptimizer().setLearningRate()

    def test(self, test_set: list[Tensor]):
        count = 0
        total = 0
        for instance in test_set:
            self.input_nodes[0].setValue(self._createInputTensor(instance))
            class_label = int(self.predict()[0])
            if class_label == instance.getValue((instance.getShape()[0] - 1,)):
                count += 1
            total += 1
        return (count + 0.0) / total

    def getOutputValue(self, output_node: ComputationalNode) -> list[float]:
        output_value = output_node.getValue()
        if output_value is None:
            return []
        cols = output_value.getShape()[1]
        max_val = float('-inf')
        label_index = -1
        for j in range(cols):
            val = output_value.getValue((0, j))
            if val > max_val:
                max_val = val
                label_index = j
        return [float(label_index)]
