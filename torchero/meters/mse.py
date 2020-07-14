import math

import torch

from torchero.meters.aggregators.batch import Average
from torchero.meters.batch import BatchMeter


class MSE(BatchMeter):
    """ Meter for mean squared error metric
    """

    DEFAULT_MODE = 'min'
    INVALID_BATCH_DIMENSION_MESSAGE = (
        'Expected both tensors have at less two dimension and same shape'
    )
    INVALID_INPUT_TYPE_MESSAGE = (
        'Expected types (FloatTensor, FloatTensor) as inputs'
    )

    def __init__(self, transform=None):
        """ Constructor
        """
        super(MSE, self).__init__(transform=transform, aggregator=Average())

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


class MSLE(MSE):
    """ Meter for mean squared log error metric
    """
    def __init__(self, transform=None):
        transform = transform or (lambda x: x)
        super(MSLE, self).__init__(transform=lambda x: torch.log(transform(x)+1))


class RMSLE(MSLE):
    """ Meter for rooted mean squared log error metric
    """
    def value(self):
        return math.sqrt(super(RMSLE, self).value())
