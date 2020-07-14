from .base import BaseMeter
from .exceptions import ZeroMeasurementsError


class NullMeter(BaseMeter):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def measure(self, *batchs):
        pass

    def reset(self):
        pass

    def clone(self):
        return self

    def value(self):
        raise ZeroMeasurementsError()

    def __repr__(self):
        return "NullMeter()"
