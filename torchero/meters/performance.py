import copy
from abc import abstractmethod
import time

from torchero.meters.base import BaseMeter
from torchero.utils import defaults

class BaseSpeedMeter(BaseMeter):
    """ Base class to measure the training performance
    """
    DEFAULT_MODE = 'max'

    def __init__(self, time_unit='seconds'):
        self.time_unit = defaults.parse_time_unit(time_unit)
        self.reset()

    @abstractmethod
    def increment_unit(self, batch):
        pass

    def measure(self, *batch):
        current_time = time.time()
        self.increment_unit(batch)
        self._running_time += current_time - self._last_time
        self._last_time = current_time

    def reset(self):
        self._unit = 0
        self._last_time = time.time()
        self._running_time = 0

    def value(self):
        return float('inf') if self._running_time == 0 else self._unit * self.time_unit/self._running_time

class BatchSpeed(BaseSpeedMeter):
    """ Measure the training performance in batches/`time_unit`
    """
    def increment_unit(self, batch):
        self._unit += 1

class IterSpeed(BaseSpeedMeter):
    """ Measure the training performance in samples/`time_unit`
    """
    def increment_unit(self, batch):
        X = next(iter(batch))
        # Asumes batch dimention is the first one
        self._unit += len(X)

class BasePaceMeter(BaseMeter):
    DEFAULT_MODE = 'min'

    def value(self):
        return 1/super(BasePaceMeter, self).value()

class BatchPace(BasePaceMeter, BatchSpeed):
    """ Measure the training pace in `time_unit`/batches
    """
    pass

class IterPace(BasePaceMeter, IterSpeed):
    """ Measure the training pace in `time_unit`/samples
    """
    pass
