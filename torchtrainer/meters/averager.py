from .base import BaseMeter
from .exceptions import ZeroMeasurementsError

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
