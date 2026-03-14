from __future__ import annotations

from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode
from ComputationalGraph.Function.Softmax import Softmax
from Math.Tensor import Tensor


class LinearPerceptronSingleInput(ComputationalGraph):

    def __init__(self, parameters: NeuralNetworkParameter):
        super().__init__(parameters)

    def _createInputTensor(self, instance: Tensor) -> Tensor:
        data = [instance.getValue((i,)) for i in range(instance.getShape()[0] - 1)]
        return Tensor(data, (1, instance.getShape()[0] - 1))

    def train(self, train_set: list[Tensor]):
        input_node = MultiplicationNode(learnable=False, is_biased=True, is_hadamard=False)
        self.input_nodes.append(input_node)

        weights_tensor = Tensor([1.0, 1.0, 1.0, 1.0], (2, 2))
        w = MultiplicationNode(value=weights_tensor)
        a = self.addEdge(input_node, w, is_biased=False)
        self.output_node = self.addEdge(a, Softmax(), is_biased=False)

        data_tensor = Tensor([1.0, 1.0], (2,))
        input_node.setValue(self._createInputTensor(data_tensor))
        self.forwardCalculation()
        self.backpropagation()

    def test(self, test_set: list[Tensor]):
        return None

    def getOutputValue(self, output_node: ComputationalNode) -> list[float]:
        return None
