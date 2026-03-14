from typing import TYPE_CHECKING
from ComputationalGraph.Optimizer.Adam import Adam
from Math.Tensor import Tensor

if TYPE_CHECKING:
    from ComputationalGraph.Node.ComputationalNode import ComputationalNode


class AdamW(Adam):
    def __init__(
        self,
        learning_rate: float,
        eta_decrease: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        weight_decay: float
    ):
        """
        Initializes the AdamW optimizer.

        :param weight_decay: The coefficient for weight decay (decoupled L2 regularization).
        """
        super().__init__(learning_rate, eta_decrease, beta1, beta2, epsilon)
        self.__weight_decay: float = weight_decay

    def setGradients(self, node: "ComputationalNode"):
        """
        Sets the gradients using the AdamW logic:
        1. Calculate the standard Adam update.
        2. Add the decoupled weight decay term: (lr * wd * weight_value).
        """
        gradients = self._calculate(node)

        current_values = node.getValue().getData()

        decay_factor = self._learning_rate * self.__weight_decay

        final_gradients = []
        for i in range(len(gradients)):
            updated_grad = gradients[i] + (decay_factor * current_values[i])
            final_gradients.append(updated_grad)

        node.setBackward(Tensor(final_gradients, node.getBackward().getShape()))