from abc import ABCMeta, abstractmethod

from torchero.meters.exceptions import ZeroMeasurementsError


class Aggregator(object, metaclass=ABCMeta):
    """ Base class for all aggregators
    All aggregators should subclass this class
    """

    def init(self):
        self._num_samples = 0
        return self.initial_value()

    @property
    def num_samples(self):
        return self._num_samples

    @abstractmethod
    def initial_value(self):
        pass

    @abstractmethod
    def combine(self, old_result, value):
        pass

    def final_value(self, result):
        if self._num_samples == 0:
            raise ZeroMeasurementsError()
        return result
