import torch
from abc import abstractmethod
from enum import Enum
from torch import nn
from .base import BaseMeter, ZeroMeasurementsError

class ResultMode(Enum):
    SUM = 1
    NORMALIZED = 2
    PERCENTAGE = 3

class _CategoricalAccuracy(BaseMeter):
    INVALID_BATCH_DIMENSION_MESSAGE = 'Expected both tensors have at less two dimension and same shape'
    INVALID_INPUT_TYPE_MESSAGE = 'Expected types (Tensor, LongTensor) as inputs'
    RESULT_MODE_ERROR_MESSAGE = ('Mode {} not recognized. Options are '
                                 'ResultMode.SUM, ResultMode.NORMALIZED, ResultMode.PERCENTAGE')

    def __init__(self, result_mode=ResultMode.NORMALIZED):
        """ Constructor

        Arguments:
            size_average (bool): Average of batch size
        """
        if not isinstance(result_mode, ResultMode):
            raise ValueError(self.RESULT_MODE_ERROR_MESSAGE.format(mode=result_mode))

        self.result_mode = result_mode
        self.reset()

    def reset(self):
        self.result = 0.0
        self.num_samples = 0

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

        self.result += torch.sum(self._get_result(a, b))
        self.num_samples += len(b)

    def value(self):
        if self.num_samples == 0:
            raise ZeroMeasurementsError()

        if self.result_mode is ResultMode.SUM:
            return self.result
        elif self.result_mode is ResultMode.NORMALIZED:
            return self.result / self.num_samples
        else:
            return self.result * 100.0 / self.num_samples

class CategoricalAccuracy(_CategoricalAccuracy):
    """ Meter for accuracy categorical on categorical targets
    """
    def _get_result(self, a, b):
        predictions = a.topk(k=1, dim=1)[1].squeeze(1)
        return (predictions == b)

class BinaryAccuracy(_CategoricalAccuracy):
    """ Meter for accuracy on binary targets (assuming normalized inputs)
    """
    def __init__(self, threshold=0.5, result_mode=ResultMode.NORMALIZED):
        """ Constructor

        Arguments:
            threshold (float): Positive/Negative class separation threshold
        """
        super(BinaryAccuracy, self).__init__(result_mode=result_mode)
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
    def __init__(self, result_mode=ResultMode.NORMALIZED, threshold=0.5, activation=None):
        super(BinaryWithLogitsAccuracy, self).__init__(threshold=threshold, result_mode=result_mode)
        self.activation = activation
        if self.activation is None:
            self.activation = nn.Sigmoid()

    def _get_result(self, output, target):
        return super(BinaryWithLogitsAccuracy, self)._get_result(self.activation(output), target)
