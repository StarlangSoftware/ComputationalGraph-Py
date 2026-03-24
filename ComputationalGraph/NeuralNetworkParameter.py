from ComputationalGraph.Function.CrossEntropyLoss import CrossEntropyLoss
from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Initialization.Initialization import Initialization
from ComputationalGraph.Initialization.RandomInitialization import RandomInitialization
from ComputationalGraph.Optimizer.Optimizer import Optimizer


class NeuralNetworkParameter:
    """
    Parameter class for neural network models.
    """

    __seed: int
    __optimizer: Optimizer
    __epoch: int
    __initialization: Initialization
    __dropout: float
    __loss_function: Function
    __batch_size: int

    def __init__(self,
                 seed: int,
                 epoch: int,
                 optimizer: Optimizer,
                 initialization: Initialization = None,
                 loss_function: Function = None,
                 dropout: float = 0.0,
                 batch_size: int = 1):
        """
        Constructor for NeuralNetworkParameter.

        :param seed: Random seed.
        :param epoch: Number of epochs.
        :param optimizer: Optimizer object.
        :param initialization: Initialization method.
        :param loss_function: Loss function.
        :param dropout: Dropout ratio.
        :param batch_size: Batch size.
        """
        self.__seed = seed
        self.__optimizer = optimizer
        self.__epoch = epoch
        self.__initialization = initialization if initialization is not None else RandomInitialization()
        self.__dropout = dropout
        self.__loss_function = loss_function if loss_function is not None else CrossEntropyLoss()
        self.__batch_size = batch_size

    def getSeed(self) -> int:
        """
        Getter for seed.

        :return: Seed value.
        """
        return self.__seed

    def getOptimizer(self) -> Optimizer:
        """
        Getter for optimizer.

        :return: Optimizer object.
        """
        return self.__optimizer

    def getEpoch(self) -> int:
        """
        Getter for epoch.

        :return: Epoch count.
        """
        return self.__epoch

    def getInitialization(self) -> Initialization:
        """
        Getter for initialization.

        :return: Initialization object.
        """
        return self.__initialization

    def initializeWeights(self, row: int, column: int, random) -> list[float]:
        """
        Initializes weights.

        :param row: Row size.
        :param column: Column size.
        :param random: Random object.
        :return: Weight list.
        """
        return self.__initialization.initialize(row, column, random)

    def getDropout(self) -> float:
        """
        Getter for dropout.

        :return: Dropout value.
        """
        return self.__dropout

    def getLossFunction(self) -> Function:
        """
        Getter for loss function.

        :return: Loss function.
        """
        return self.__loss_function

    def getBatchSize(self) -> int:
        """
        Getter for batch size.

        :return: Batch size.
        """
        return self.__batch_size