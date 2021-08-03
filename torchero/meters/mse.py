import math

import torch

from torchero.meters.aggregators.batch import Average, Maximum
from torchero.meters.batch import BatchMeter

class RegressionMetric(BatchMeter):
    """ Meter for mean squared error metric
    """
    DEFAULT_MODE = 'min'
    INVALID_BATCH_DIMENSION_MESSAGE = (
        'Expected both tensors have at less two dimension and same shape'
    )
    INVALID_INPUT_TYPE_MESSAGE = (
        'Expected types (FloatTensor, FloatTensor) as inputs'
    )
    agg_func = Average

    def __init__(self, transform=None):
        """ Constructor
        """
        super(RegressionMetric, self).__init__(transform=transform, aggregator=self.agg_func())

    def measure(self, a, b):
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if len(a.size()) != 2 or b.shape != a.shape:
            raise ValueError(self.INVALID_BATCH_DIMENSION_MESSAGE)
        super(RegressionMetric, self).measure(a, b)

class MAE(RegressionMetric):
    """ Meter for mean absolute error metric
    """
    name = 'mae'
    def _get_result(self, a, b):
        return (a-b).abs()


class MSE(RegressionMetric):
    """ Meter for mean squared error metric
    """
    name = 'mse'
    def _get_result(self, a, b):
        return torch.pow(a-b, 2)


class RMSE(MSE):
    """ Meter for rooted mean squared error metric
    """
    name = 'rmse'
    def value(self):
        return math.sqrt(super(RMSE, self).value())


class MSLE(RegressionMetric):
    """ Meter for mean squared log error metric
    """
    name = 'msle'
    def _get_result(self, a, b):
        return torch.pow(torch.log(a+1)-torch.log(b+1), 2)


class RMSLE(MSLE):
    """ Meter for rooted mean squared log error metric
    """
    name = 'rmsle'
    def value(self):
        return math.sqrt(super(RMSLE, self).value())
