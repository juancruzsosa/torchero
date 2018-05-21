import torch
import math
import math
from .batch import BatchMeter
from .exceptions import ZeroMeasurementsError

class MSE(BatchMeter):
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
        super(MSE, self).__init__()
        self.take_sqrt = take_sqrt

    def _get_result(self, a, b):
        return torch.pow(a-b, 2)

    def measure(self, a, b):
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if len(a.size()) != 2 or b.shape != a.shape:
            raise ValueError(self.INVALID_BATCH_DIMENSION_MESSAGE)
        super(MSE, self).measure(a, b)

    def value(self):
        result = super(MSE, self).value()
        if self.take_sqrt:
            result = math.sqrt(result)
        return result
