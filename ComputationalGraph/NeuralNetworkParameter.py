from ComputationalGraph.Function.CrossEntropyLoss import CrossEntropyLoss
from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Initialization.Initialization import Initialization
from ComputationalGraph.Initialization.RandomInitialization import RandomInitialization
from ComputationalGraph.Optimizer.Optimizer import Optimizer


class NeuralNetworkParameter:

    def __init__(self, seed: int, epoch: int, optimizer: Optimizer,
                 initialization: Initialization = None,
                 loss_function: Function = None,
                 dropout: float = 0.0,
                 batch_size: int = 1):
        self.seed = seed
        self.optimizer = optimizer
        self.epoch = epoch
        self.initialization = initialization if initialization is not None else RandomInitialization()
        self.dropout = dropout
        self.loss_function = loss_function if loss_function is not None else CrossEntropyLoss()
        self.batch_size = batch_size

    def getOptimizer(self) -> Optimizer:
        return self.optimizer

    def getEpoch(self) -> int:
        return self.epoch

    def initializeWeights(self, row: int, column: int, random) -> list[float]:
        return self.initialization.initialize(row, column, random)

    def getDropout(self) -> float:
        return self.dropout

    def getLossFunction(self) -> Function:
        return self.loss_function

    def getBatchSize(self) -> int:
        return self.batch_size