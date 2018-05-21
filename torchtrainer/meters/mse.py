import torch
import math
import math
from .base import BaseMeter 
from .exceptions import ZeroMeasurementsError

class MSE(BaseMeter):
    """ Meter for mean squared error metric
    """

    INVALID_BATCH_DIMENSION_MESSAGE = ('Expected both tensors have at less two '
                                       'dimension and same shape')
    INVALID_INPUT_TYPE_MESSAGE = ('Expected types (FloatTensor, FloatTensor) '
                                  'as inputs')

    def __init__(self, take_sqrt=False):
        """ Constructor

        Arguments:
            take_sqrt (bool): Take square root in final results
        """
        self.take_sqrt = take_sqrt
        self.reset()

    def reset(self):
        self.result = 0.0
        self.num_samples = 0

    def _get_result(self, a, b):
        return torch.pow(a-b, 2)

    def measure(self, a, b):
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if len(a.size()) != 2 or b.shape != a.shape:
            raise ValueError(self.INVALID_BATCH_DIMENSION_MESSAGE)

        self.result += torch.sum(self._get_result(a, b))
        self.num_samples += len(b)

    def value(self):
        if self.num_samples == 0:
            raise ZeroMeasurementsError()

        result = self.result / self.num_samples
        if self.take_sqrt:
            result = math.sqrt(result)
        return result
