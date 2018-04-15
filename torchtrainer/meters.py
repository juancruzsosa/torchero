import torch
from abc import abstractmethod
from enum import Enum

class BaseMeter(object):
    """ Interface for all meters.
    All meters should subclass this class
    """
    @abstractmethod
    def measure(self, *batchs):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def value(self):
        pass

class ZeroMeasurementsError(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return "No measurements has been made"

class ResultMode(Enum):
    SUM = 1
    NORMALIZED = 2
    PERCENTAGE = 3

class CategoricalAccuracy(BaseMeter):
    """ Meter of Categorical accuracy (for more than two classes)
    """

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

    def measure(self, a, b):
        if not torch.is_tensor(a):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if not isinstance(b, torch.LongTensor) and not isinstance(b, torch.cuda.LongTensor):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if len(a.size()) != 2 or len(b.size()) != 1 or len(b) != a.size()[0]:
            raise ValueError(self.INVALID_BATCH_DIMENSION_MESSAGE)

        predictions = a.topk(k=1, dim=1)[1].squeeze(1)
        self.result += torch.sum(predictions == b)

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

class Averager(BaseMeter):
    """ Meter that returns the average over all measured values
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.result = None
        self.num_samples = 0

    def measure(self, value):
        if self.result is None:
            self.result = value
        else:
            self.result += value

        self.num_samples += 1

    def value(self):
        if self.num_samples == 0:
            raise ZeroMeasurementsError()

        return self.result / self.num_samples
