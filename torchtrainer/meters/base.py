from abc import abstractmethod, ABCMeta
import copy

class ZeroMeasurementsError(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return "No measurements has been made"

class BaseMeter(object, metaclass=ABCMeta):
    """ Interface for all meters.
    All meters should subclass this class
    """
    @abstractmethod
    def measure(self, *batchs):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def value(self):
        pass

    def clone(self):
        return copy.deepcopy(self)
