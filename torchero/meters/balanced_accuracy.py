from collections import Counter

import torch

from torchero.meters.base import BaseMeter


class BalancedAccuracy(BaseMeter):
    """ Calculate the balanced categorical accuracy on categorical targets
    to deal with imbalanced datasets. The metric is equivalent to the average
    of recalls on each class
    """
    name = 'balanced_acc'
    DEFAULT_MODE = 'max'

    def __init__(self):
        super(BalancedAccuracy, self).__init__()
        self.reset()

    def reset(self):
        self.totals_counter = Counter()
        self.match_counter = Counter()

    def check_tensors(self, a, b):
        if not torch.is_tensor(a):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if (not isinstance(b, torch.LongTensor) and
                not isinstance(b, torch.cuda.LongTensor)):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if len(a.size()) != 2 or len(b.size()) != 1 or len(b) != a.size()[0]:
            raise ValueError(self.INVALID_BATCH_DIMENSION_MESSAGE)

    def measure(self, a, b):
        self.check_tensors(a, b)

        if a.dim() == 2:
            a = a.argmax(dim=1)

        if len(a) != len(b):
            raise Exception(self.INVALID_LENGTHS_MESSAGE)

        self.totals_counter.update(b.cpu().tolist())
        self.match_counter.update(b[b == a].cpu().tolist())

    def value(self):
        arr = [self.match_counter.get(c, 0)/self.totals_counter[c]
               for c in self.totals_counter.keys()]
        return torch.Tensor(arr).mean().item()

    def __getstate__(self):
        return None

    def __setstate__(self, _):
        self.reset()
