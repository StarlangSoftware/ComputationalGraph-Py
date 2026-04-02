from typing import Dict, List, TYPE_CHECKING

from ComputationalGraph.Optimizer.Optimizer import Optimizer
from Math.Tensor import Tensor

if TYPE_CHECKING:
    from ComputationalGraph.Node.ComputationalNode import ComputationalNode


class SGDMomentum(Optimizer):
    """
    Stochastic Gradient Descent optimizer with momentum.
    """

    _velocity_map: Dict["ComputationalNode", List[float]]
    _momentum: float

    def __init__(self, learning_rate: float, eta_decrease: float, momentum: float) -> None:
        """
        Initializes the SGD with Momentum optimizer.

        :param learning_rate: The step size for updates.
        :param eta_decrease: The factor by which learning rate is multiplied over time.
        :param momentum: The friction/velocity coefficient.
        """
        super().__init__(learning_rate, eta_decrease)

        self._velocity_map = {}
        self._momentum = momentum

    def setGradients(self, node: "ComputationalNode") -> None:
        """
        Calculates the new gradients by combining the current gradient with the previous velocity.

        :param node: Computational node whose gradients will be updated.
        """
        backward_data = node.getBackward().getData()
        backward_size = len(backward_data)

        new_values = [(1 - self._momentum) * val for val in backward_data]

        if node in self._velocity_map:
            velocity = self._velocity_map[node]
            for i in range(backward_size):
                new_values[i] += velocity[i] * self._momentum

        self._velocity_map[node] = list(new_values)

        final_update = [val * self._learning_rate for val in new_values]

        node.setBackward(Tensor(final_update, node.getBackward().getShape()))