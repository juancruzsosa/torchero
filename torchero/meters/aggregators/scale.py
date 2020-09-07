from torchero.meters.aggregators.base import Aggregator


class Scale(Aggregator):
    def __init__(self, aggregator, scale=1):
        self.aggregator = aggregator
        self.scale = scale

    def initial_value(self):
        return self.aggregator.init()

    def combine(self, old_result, result):
        result = self.aggregator.combine(old_result, result)
        self._num_samples += self.aggregator._num_samples
        return result

    def final_value(self, value):
        return self.aggregator.final_value(value) * self.scale

    def __repr__(self):
        return "{}*{}".format(repr(self.aggregator), repr(self.scale))


def percentage(aggregator):
    return Scale(aggregator, scale=100.0)
