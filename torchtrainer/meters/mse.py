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

    def __init__(self):
        """ Constructor
        """
        super(MSE, self).__init__()

    def _get_result(self, a, b):
        return torch.pow(a-b, 2)

    def measure(self, a, b):
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if len(a.size()) != 2 or b.shape != a.shape:
            raise ValueError(self.INVALID_BATCH_DIMENSION_MESSAGE)
        super(MSE, self).measure(a, b)

class RMSE(MSE):
    """ Meter for rooted mean squared error metric
    """
    def value(self):
        return math.sqrt(super(RMSE, self).value())
