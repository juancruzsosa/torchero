import torch

from torchero.meters.aggregators.base import Aggregator


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


class Average(BatchAggregator):
    """ Aggregator that return the sums of all given batchs
    """
    def initial_value(self):
        return 0

    def combine_batch(self, old_result, value):
        return old_result + torch.sum(value, dim=0)

    """ Aggregator that returns the average of all given batchs
    """
    def final_value(self, result):
        result = super(Average, self).final_value(result)
        return torch.sum(result) / (self._num_samples * result.nelement())


class Maximum(BatchAggregator):
    """ Aggregator that returns the maximum of all given batchs
    """
    def initial_value(self):
        return None

    def combine_batch(self, old_result, value):
        value = torch.max(value)
        return value if old_result is None else max(old_result, value)


class Minimum(BatchAggregator):
    """ Aggregator that returns the minimum of all given batchs
    """
    def initial_value(self):
        return None

    def combine_batch(self, old_result, value):
        value = torch.min(value)
        return value if old_result is None else min(old_result, value)
