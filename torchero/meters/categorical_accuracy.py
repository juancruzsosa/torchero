import torch

from .batch import BatchMeter


class CategoricalAccuracy(BatchMeter):
    """ Accuracy on categorical targets
    """

    name = 'acc'
    DEFAULT_MODE = 'max'
    INVALID_BATCH_DIMENSION_MESSAGE = (
        'Expected both tensors have at less two dimension and same shape'
    )
    INVALID_INPUT_TYPE_MESSAGE = (
        'Expected types (Tensor, LongTensor) as inputs'
    )

    def __init__(self, k=1, aggregator=None):
        super(CategoricalAccuracy, self).__init__(aggregator=aggregator)
        self.k = k

    def check_tensors(self, a, b):
        if not torch.is_tensor(a):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if (not isinstance(b, torch.LongTensor) and
                not isinstance(b, torch.cuda.LongTensor)):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if len(a.size()) != 2 or len(b.size()) != 1 or len(b) != a.size()[0]:
            raise ValueError(self.INVALID_BATCH_DIMENSION_MESSAGE)

    def _get_result(self, a, b):
        return torch.sum((a.topk(k=self.k, dim=1)[1] ==
                          b.unsqueeze(-1)).float(), dim=1)
