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
        """ Nunmber of samples
        """
        return self._num_samples

    @abstractmethod
    def initial_value(self):
        """ Neutral value
        """
        pass

    @abstractmethod
    def combine(self, old_result, value):
        """ Combines the metrics of the lasts batch with the current one

        Arguments:
            old_result: Accumulated values of the previous batches
            value: Values for this batch
        """
        pass

    def final_value(self, result):
        """ Returns the final metric value given the accumulated values
        """
        if self._num_samples == 0:
            raise ZeroMeasurementsError()
        return result

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)
