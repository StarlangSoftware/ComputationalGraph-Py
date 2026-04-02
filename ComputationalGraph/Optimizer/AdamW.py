from typing import TYPE_CHECKING

from ComputationalGraph.Optimizer.Adam import Adam
from Math.Tensor import Tensor

if TYPE_CHECKING:
    from ComputationalGraph.Node.ComputationalNode import ComputationalNode


class AdamW(Adam):
    """
    AdamW optimizer implementation.
    """

    __weight_decay: float

    def __init__(
        self,
        learning_rate: float,
        eta_decrease: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        weight_decay: float
    ) -> None:
        """
        Initializes the AdamW optimizer.

        :param learning_rate: Step size.
        :param eta_decrease: Learning rate decay factor.
        :param beta1: Exponential decay rate for the first moment.
        :param beta2: Exponential decay rate for the second moment.
        :param epsilon: Small constant for numerical stability.
        :param weight_decay: Coefficient for weight decay.
        """
        super().__init__(learning_rate, eta_decrease, beta1, beta2, epsilon)
        self.__weight_decay = weight_decay

    def setGradients(self, node: "ComputationalNode") -> None:
        """
        Sets the gradients using the AdamW update rule.

        :param node: Computational node whose gradients will be updated.
        """
        gradients = self._calculate(node)
        current_values = node.getValue().getData()
        decay_factor = self._learning_rate * self.__weight_decay

        final_gradients = []
        for i in range(len(gradients)):
            updated_grad = gradients[i] + (decay_factor * current_values[i])
            final_gradients.append(updated_grad)

        node.setBackward(Tensor(final_gradients, node.getBackward().getShape()))