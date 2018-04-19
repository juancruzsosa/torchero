from .base import Aggregator

class Scale(Aggregator):
    def __init__(self, aggregator, scale=1):
        self.aggregator = aggregator
        self.scale = scale

    def init(self):
        return self.aggregator.init()

    def combine(self, old_result, result):
        return self.aggregator.combine(old_result, result)

    def final_value(self, value):
        return self.aggregator.final_value(value) * self.scale

def percentage(aggregator):
    return Scale(aggregator, scale=100.0)
