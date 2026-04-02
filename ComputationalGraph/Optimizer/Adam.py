import math
from typing import List, Dict, TYPE_CHECKING

from ComputationalGraph.Optimizer.SGDMomentum import SGDMomentum
from Math.Tensor import Tensor

if TYPE_CHECKING:
    from ComputationalGraph.Node.ComputationalNode import ComputationalNode


class Adam(SGDMomentum):
    """
    Adam optimizer implementation.
    """

    __momentum_map: Dict["ComputationalNode", List[float]]
    __beta2: float
    __epsilon: float
    __current_beta1: float
    __current_beta2: float

    def __init__(self,
                 learning_rate: float,
                 eta_decrease: float,
                 beta1: float,
                 beta2: float,
                 epsilon: float) -> None:
        """
        Initializes the Adam optimizer.

        :param learning_rate: Step size.
        :param eta_decrease: Learning rate decay factor.
        :param beta1: Exponential decay rate for the first moment.
        :param beta2: Exponential decay rate for the second moment.
        :param epsilon: Small constant for numerical stability.
        """
        super().__init__(learning_rate, eta_decrease, beta1)

        self.__momentum_map = {}
        self.__beta2 = beta2
        self.__epsilon = epsilon
        self.__current_beta1 = 1.0
        self.__current_beta2 = 1.0

    def _calculate(self, node: "ComputationalNode") -> List[float]:
        """
        Calculates the adaptive gradient updates using the Adam formula.

        :param node: Computational node whose gradients will be updated.
        :return: Calculated update values.
        """
        backward_data = node.getBackward().getData()
        backward_size = len(backward_data)

        beta1 = self._momentum

        new_values_momentum = [(1 - beta1) * g for g in backward_data]
        new_values_velocity = [(1 - self.__beta2) * (g * g) for g in backward_data]

        if node in self.__momentum_map:
            hist_m = self.__momentum_map[node]
            hist_v = self._velocity_map[node]

            for i in range(backward_size):
                new_values_momentum[i] += beta1 * hist_m[i]
                new_values_velocity[i] += self.__beta2 * hist_v[i]

        self.__momentum_map[node] = list(new_values_momentum)
        self._velocity_map[node] = list(new_values_velocity)

        m_corr = 1 - self.__current_beta1
        v_corr = 1 - self.__current_beta2

        final_updates: List[float] = []
        for i in range(backward_size):
            m_hat = new_values_momentum[i] / m_corr
            v_hat = new_values_velocity[i] / v_corr

            adaptive_lr = (m_hat / (math.sqrt(v_hat) + self.__epsilon)) * self._learning_rate
            final_updates.append(adaptive_lr)

        return final_updates

    def setGradients(self, node: "ComputationalNode") -> None:
        """
        Sets the backward tensor using the Adam calculation.

        :param node: Computational node whose backward tensor will be updated.
        """
        update_values = self._calculate(node)
        node.setBackward(Tensor(update_values, node.getBackward().getShape()))

    def updateValues(self, leaf_nodes: List["ComputationalNode"]) -> None:
        """
        Updates the bias correction coefficients and triggers recursive parameter updates.

        :param leaf_nodes: Leaf nodes to be updated.
        """
        self.__current_beta1 *= self._momentum
        self.__current_beta2 *= self.__beta2

        super().updateValues(leaf_nodes)