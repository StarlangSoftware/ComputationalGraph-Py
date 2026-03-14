from typing import List, TYPE_CHECKING
from ComputationalGraph.Optimizer.Optimizer import Optimizer
from Math.Tensor import Tensor

if TYPE_CHECKING:
    from ComputationalGraph.Node.ComputationalNode import ComputationalNode


class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate: float, eta_decrease: float):
        """
        Initializes the Stochastic Gradient Descent (SGD) optimizer.

        :param learning_rate: The step size for updates.
        :param eta_decrease: The factor by which learning rate is multiplied over time.
        """
        super().__init__(learning_rate, eta_decrease)

    def setGradients(self, node: "ComputationalNode"):
        """
        Sets the gradients (backward values) of the node to the learning rate
        times the current backward values.
        """
        # Retrieve the raw gradient data
        backward_data = node.getBackward().getData()

        # Scale each gradient value by the current learning rate
        scaled_values = [val * self._learning_rate for val in backward_data]

        # Update the node's backward tensor with the scaled gradients
        node.setBackward(Tensor(scaled_values, node.getBackward().getShape()))