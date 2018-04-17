from .base import BaseMeter, ZeroMeasurementsError

class NullMeter(BaseMeter):
    def measure(self, *batchs):
        pass

    def reset(self):
        pass

    def value(self):
        raise ZeroMeasurementsError()
