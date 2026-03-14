import math
from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor

class Softmax(Function):
    """
    Softmax activation function representation for the computational graph.
    Computes probabilities across the last dimension of the tensor.
    """

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Computes the Softmax activation for the given tensor.

        :param tensor: The input tensor.
        :return: A new Tensor containing Softmax(x).
        """
        old_values = tensor.getData()
        shape = tensor.getShape()
        last_dim_size = shape[-1]

        new_data = []

        # Process the flat array in chunks of 'last_dim_size'
        for i in range(0, len(old_values), last_dim_size):
            chunk = old_values[i: i + last_dim_size]

            # Calculate exponentials for the chunk
            exp_chunk = [math.exp(val) for val in chunk]
            chunk_sum = sum(exp_chunk)

            # Divide each exponential by the sum of the chunk and add to our new data
            new_data.extend([exp_val / chunk_sum for exp_val in exp_chunk])

        return Tensor(new_data, shape)

    def derivative(self, tensor: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the Softmax activation function.
        Note: 'tensor' is the output of the Softmax forward pass (y).

        :param tensor: The output tensor of the Softmax(x) operation.
        :param backward: The backward gradient tensor.
        :return: A new Tensor containing the gradient values.
        """
        old_values = tensor.getData()
        backward_values = backward.getData()
        shape = tensor.getShape()
        last_dim_size = shape[-1]

        new_values = []

        # Process in chunks just like the forward pass
        for i in range(0, len(old_values), last_dim_size):
            y_chunk = old_values[i: i + last_dim_size]
            b_chunk = backward_values[i: i + last_dim_size]

            # Calculate the dot product of the output (y) and backward gradient for this chunk
            total = sum(y * b for y, b in zip(y_chunk, b_chunk))

            # Subtract the total from each backward value in the chunk
            new_values.extend([b - total for b in b_chunk])

        # Create a temporary tensor for the (backward - total) values
        temp_tensor = Tensor(new_values, shape)

        # Multiply element-wise by the original softmax output (y)
        return tensor.hadamardProduct(temp_tensor)

    def addEdge(self, input_nodes: List[ComputationalNode], is_biased: bool) -> ComputationalNode:
        """
        Adds a Softmax node to the computational graph.
        """
        new_node = FunctionNode(function=self, is_biased=is_biased)
        input_nodes[0].add(new_node)
        return new_node