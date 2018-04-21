import torch
from abc import abstractmethod
from enum import Enum
from torch import nn
from .base import BaseMeter, ZeroMeasurementsError
from .aggregators.batch import Average

class _CategoricalAccuracy(BaseMeter):
    INVALID_BATCH_DIMENSION_MESSAGE = 'Expected both tensors have at less two dimension and same shape'
    INVALID_INPUT_TYPE_MESSAGE = 'Expected types (Tensor, LongTensor) as inputs'
    RESULT_MODE_ERROR_MESSAGE = ('Mode {} not recognized. Options are '
                                 'ResultMode.SUM, ResultMode.NORMALIZED, ResultMode.PERCENTAGE')

    def __init__(self, aggregator=None):
        """ Constructor

        Arguments:
            size_average (bool): Average of batch size
        """
        self.aggregator = aggregator

        if self.aggregator is None:
            self.aggregator = Average()

        self.reset()

    def reset(self):
        self.result = self.aggregator.init()

    @abstractmethod
    def _get_result(self, a, b):
        pass

    def measure(self, a, b):
        if not torch.is_tensor(a):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if not isinstance(b, torch.LongTensor) and not isinstance(b, torch.cuda.LongTensor):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if len(a.size()) != 2 or len(b.size()) != 1 or len(b) != a.size()[0]:
            raise ValueError(self.INVALID_BATCH_DIMENSION_MESSAGE)

        self.result = self.aggregator.combine(self.result, self._get_result(a, b))

    def value(self):
        return self.aggregator.final_value(self.result)

class CategoricalAccuracy(_CategoricalAccuracy):
    """ Meter for accuracy categorical on categorical targets
    """
    def __init__(self, k=1, aggregator=None):
        super(CategoricalAccuracy, self).__init__(aggregator=aggregator)
        self.k = k

    def _get_result(self, a, b):
        return (a.topk(k=self.k, dim=1)[1] == b.unsqueeze(-1))

class BinaryAccuracy(_CategoricalAccuracy):
    """ Meter for accuracy on binary targets (assuming normalized inputs)
    """
    def __init__(self, threshold=0.5, aggregator=None):
        """ Constructor

        Arguments:
            threshold (float): Positive/Negative class separation threshold
        """
        super(BinaryAccuracy, self).__init__(aggregator=aggregator)
        self.threshold = threshold

    def _get_result(self, output, target):
        predictions = output.squeeze(1) >= self.threshold
        if predictions.is_cuda:
            predictions = predictions.type(torch.cuda.LongTensor)
        else:
            predictions = predictions.type(torch.LongTensor)
        return (predictions == target)

class BinaryWithLogitsAccuracy(BinaryAccuracy):
    """ Binary accuracy meter with an integrated activation function
    """
    def __init__(self, aggregator=None, threshold=0.5, activation=None):
        super(BinaryWithLogitsAccuracy, self).__init__(threshold=threshold, aggregator=aggregator)
        self.activation = activation
        if self.activation is None:
            self.activation = nn.Sigmoid()

    def _get_result(self, output, target):
        return super(BinaryWithLogitsAccuracy, self)._get_result(self.activation(output), target)
