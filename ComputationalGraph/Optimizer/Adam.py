import math
from typing import List, Dict, Set, TYPE_CHECKING
from ComputationalGraph.Optimizer.SGDMomentum import SGDMomentum
from Math.Tensor import Tensor

if TYPE_CHECKING:
    from ComputationalGraph.Node.ComputationalNode import ComputationalNode


class Adam(SGDMomentum):
    def __init__(self, learning_rate: float, eta_decrease: float, beta1: float, beta2: float, epsilon: float):
        """
        Initializes the Adam optimizer.

        :param learning_rate: Step size.
        :param eta_decrease: Learning rate decay factor.
        :param beta1: Exponential decay rate for the first moment (momentum).
        :param beta2: Exponential decay rate for the second moment (velocity).
        :param epsilon: Small constant for numerical stability.
        """
        super().__init__(learning_rate, eta_decrease, beta1)

        self.__momentum_map: Dict["ComputationalNode", List[float]] = {}

        self.__beta2: float = beta2
        self.__epsilon: float = epsilon

        self.__current_beta1: float = 1.0
        self.__current_beta2: float = 1.0

    def _calculate(self, node: "ComputationalNode") -> List[float]:
        """
        Calculates the adaptive gradient updates using the Adam formula.
        """
        backward_data = node.getBackward().getData()
        backward_size = len(backward_data)

        beta1 = self._momentum

        # 1. Calculate current weighted moments
        # m_t = (1 - beta1) * g_t
        new_values_momentum = [(1 - beta1) * g for g in backward_data]
        # v_t = (1 - beta2) * g_t^2
        new_values_velocity = [(1 - self.__beta2) * (g * g) for g in backward_data]

        # 2. Add decayed historical moments if they exist
        if node in self.__momentum_map:
            hist_m = self.__momentum_map[node]
            hist_v = self._velocity_map[node]  # Accessing protected velocity_map from parent

            for i in range(backward_size):
                new_values_momentum[i] += beta1 * hist_m[i]
                new_values_velocity[i] += self.__beta2 * hist_v[i]

        # 3. State Update: Save raw moments into history
        self.__momentum_map[node] = list(new_values_momentum)
        self._velocity_map[node] = list(new_values_velocity)

        # 4. Bias Correction: accounts for moments being initialized at zero
        # m_hat = m_t / (1 - beta1^t)
        # v_hat = v_t / (1 - beta2^t)
        m_corr = 1 - self.__current_beta1
        v_corr = 1 - self.__current_beta2

        # 5. Final Adaptive Update
        # update = (m_hat / (sqrt(v_hat) + eps)) * lr
        final_updates = []
        for i in range(backward_size):
            m_hat = new_values_momentum[i] / m_corr
            v_hat = new_values_velocity[i] / v_corr

            adaptive_lr = (m_hat / (math.sqrt(v_hat) + self.__epsilon)) * self._learning_rate
            final_updates.append(adaptive_lr)

        return final_updates

    def setGradients(self, node: "ComputationalNode"):
        """Sets the backward tensor using the Adam calculation."""
        update_values = self._calculate(node)
        node.setBackward(Tensor(update_values, node.getBackward().getShape()))

    def updateValues(self, leaf_nodes: List["ComputationalNode"]):
        """Updates the bias correction coefficients and triggers the recursive update."""
        # Update power of betas (t -> t+1)
        self.__current_beta1 *= self._momentum
        self.__current_beta2 *= self.__beta2

        # Call Optimizer.update_values (via SGDMomentum)
        super().updateValues(leaf_nodes)