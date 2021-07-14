from abc import abstractmethod

import torch

from torchero.meters.aggregators.batch import Average
from torchero.meters.base import BaseMeter


class BatchMeter(BaseMeter):
    INVALID_INPUT_TYPE_MESSAGE = (
        'Expected types (Tensor, LongTensor) as inputs'
    )

    def __init__(self, aggregator=None, transform=None):
        """ Constructor

        Arguments:
            size_average (bool): Average of batch size
        """
        self.aggregator = aggregator
        self.scale = 1
        self._transform = transform

        if self.aggregator is None:
            self.aggregator = Average()

        self.reset()

    def reset(self):
        self.result = self.aggregator.init()

    @abstractmethod
    def _get_result(self, *xs):
        pass

    def check_tensors(self, *xs):
        pass

    def measure(self, *xs):
        self.check_tensors(*xs)
        if self._transform is not None:
            xs = map(self._transform, xs)
        self.result = self.aggregator.combine(self.result,
                                              self._get_result(*xs))

    def value(self):
        val = self.aggregator.final_value(self.result) * self.scale
        if torch.is_tensor(val) and val.dim() == 0:
            return val.item()
        else:
            return val

    def __mul__(self, y):
        self.scale *= y
        return self
