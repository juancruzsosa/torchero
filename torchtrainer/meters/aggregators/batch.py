import torch
from .base import Aggregator

class BatchAggregator(Aggregator):
    def combine(self, old_result, batch):
        self._num_samples += len(batch)
        return self.combine_batch(old_result, batch)

class Sum(BatchAggregator):
    """ Aggregator that return the sums of all given batchs
    """
    def initial_value(self):
        return 0

    def combine_batch(self, old_result, value):
        return old_result + torch.sum(value)

class Average(Sum):
    """ Aggregator that returns the average of all given batchs
    """
    def final_value(self, result):
        return super(Average, self).final_value(result) / self.num_samples
