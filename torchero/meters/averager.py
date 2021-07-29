import torch

from torchero.meters.base import BaseMeter
from torchero.meters.exceptions import ZeroMeasurementsError


class Averager(BaseMeter):
    """ Averages model outputs
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

        result = self.result

        if torch.is_tensor(result) and result.dim() == 0:
            result = result.item()

        result = result / self.num_samples

        return result
