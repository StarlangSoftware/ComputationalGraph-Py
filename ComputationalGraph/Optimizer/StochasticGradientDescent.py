from typing import TYPE_CHECKING

from ComputationalGraph.Optimizer.Optimizer import Optimizer
from Math.Tensor import Tensor

if TYPE_CHECKING:
    from ComputationalGraph.Node.ComputationalNode import ComputationalNode


class StochasticGradientDescent(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    """

    def __init__(self, learning_rate: float, eta_decrease: float) -> None:
        """
        Initializes the Stochastic Gradient Descent optimizer.

        :param learning_rate: The step size for updates.
        :param eta_decrease: Learning rate decay factor.
        """
        super().__init__(learning_rate, eta_decrease)

    def setGradients(self, node: "ComputationalNode") -> None:
        """
        Sets the gradients of the given node using SGD.

        :param node: Computational node whose gradients will be updated.
        """
        backward_data = node.getBackward().getData()
        scaled_values = [val * self._learning_rate for val in backward_data]
        node.setBackward(Tensor(scaled_values, node.getBackward().getShape()))