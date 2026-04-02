import unittest
import csv
import random
from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode
from ComputationalGraph.Function.CrossEntropyLoss import CrossEntropyLoss
from ComputationalGraph.Optimizer.StochasticGradientDescent import StochasticGradientDescent
from Math.Tensor import Tensor

from test.LinearPerceptronSingleInput import LinearPerceptronSingleInput
from test.NeuralNet import NeuralNet


class ComputationalGraphTest(unittest.TestCase):

    def testLinearPerceptronSingleInput(self):
        graph = LinearPerceptronSingleInput(
            NeuralNetworkParameter(1, 100, StochasticGradientDescent(0.1, 0.99))
        )
        graph.train([])

    def testNeuralNet(self):
        label_map = {}
        data_set = []

        with open("iris.txt", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                data_set.append(row)
                label = row[-1]
                if label not in label_map:
                    label_map[label] = len(label_map)

        rng = random.Random(1)
        rng.shuffle(data_set)

        train_list = []
        test_list = []
        for i, row in enumerate(data_set):
            values = [float(v) for v in row[:-1]]
            values.append(float(label_map[row[-1]]))
            tensor = Tensor(values, (len(values),))
            if i >= 120:
                test_list.append(tensor)
            else:
                train_list.append(tensor)

        graph = NeuralNet(
            NeuralNetworkParameter(1, 100, StochasticGradientDescent(0.1, 0.99),
                                   loss_function=CrossEntropyLoss(), dropout=0.0)
        )
        graph.train(train_list)
        accuracy = graph.test(test_list)
        print(f"Accuracy: {accuracy}")
        self.assertAlmostEqual(0.90, accuracy, delta=0.08)

    def testFeatures(self):

        class TestGraph(ComputationalGraph):
            def train(self_inner, train_set):
                input_node = MultiplicationNode(learnable=False, is_biased=False)
                self_inner.input_nodes.append(input_node)
                input_node.setValue(Tensor([1.0, 2.0, 3.0, 4.0], (2, 1, 2)))

                nodes = []
                for _ in range(4):
                    w = MultiplicationNode(value=Tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], (1, 2, 3)))
                    nodes.append(self_inner.addEdge(input_node, w))

                c = self_inner.concatEdges(nodes, 1)
                w_out = MultiplicationNode(value=Tensor([6.0, 5.0, 1.0], (1, 3, 1)))
                self_inner.output_node = self_inner.addEdge(c, w_out)

                self_inner.forwardCalculation()
                self_inner.backpropagation()

                input_node.setValue(Tensor([4.0, 3.0, 2.0, 1.0], (2, 1, 2)))
                self_inner.forwardCalculation()

                output = list(self_inner.output_node.getValue().getData())
                expected = [
                    2202.44, 2202.44, 2202.44, 2202.44,
                    973.6400000000001, 973.6400000000001, 973.6400000000001, 973.6400000000001,
                ]
                self.assertEqual(expected, output)

            def test(self_inner, test_set):
                return None

            def getOutputValue(self_inner, output_node):
                return None

        graph = TestGraph(
            NeuralNetworkParameter(1, 1, StochasticGradientDescent(0.1, 0.99))
        )
        graph.train([])


if __name__ == "__main__":
    unittest.main()