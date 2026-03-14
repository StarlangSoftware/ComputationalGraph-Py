from typing import Dict, List, TYPE_CHECKING
from ComputationalGraph.Optimizer.Optimizer import Optimizer
from Math.Tensor import Tensor

if TYPE_CHECKING:
    from ComputationalGraph.Node.ComputationalNode import ComputationalNode

class SGDMomentum(Optimizer):
    def __init__(self, learning_rate: float, eta_decrease: float, momentum: float):
        """
        Initializes the SGD with Momentum optimizer.

        :param learning_rate: The step size for updates.
        :param eta_decrease: The factor by which learning rate is multiplied over time.
        :param momentum: The friction/velocity coefficient (usually 0.9).
        """
        super().__init__(learning_rate, eta_decrease)

        self._velocity_map: Dict["ComputationalNode", List[float]] = {}
        self._momentum: float = momentum

    def setGradients(self, node: "ComputationalNode"):
        """
        Calculates the new gradients by combining the current gradient with the previous velocity.
        Updates the internal velocity state and modifies the node's backward tensor.
        """
        backward_data = node.getBackward().getData()
        backward_size = len(backward_data)

        # Step 1: Calculate (1 - momentum) * current_gradient
        new_values = [(1 - self._momentum) * val for val in backward_data]

        # Step 2: Add (velocity * momentum) if a velocity exists for this node
        if node in self._velocity_map:
            velocity = self._velocity_map[node]
            for i in range(backward_size):
                new_values[i] += velocity[i] * self._momentum

        # Step 3: Update the velocity map with the new velocity (pre-learning rate adjustment)
        # Note: We create a copy of new_values to store as the velocity
        self._velocity_map[node] = list(new_values)

        # Step 4: Scale the update by the learning rate
        final_update = [val * self._learning_rate for val in new_values]

        # Step 5: Update the node's backward tensor with the momentum-adjusted values
        node.setBackward(Tensor(final_update, node.getBackward().getShape()))