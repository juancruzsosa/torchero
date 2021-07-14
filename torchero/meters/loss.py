import torch
from torch import nn
from torch.autograd import Variable

from torchero.meters.averager import Averager


class LossMeter(Averager):
    name = 'loss'
    DEFAULT_MODE = 'min'

    def __init__(self, criterion):
        super(LossMeter, self).__init__()
        self.criterion = criterion

    def measure(self, x, y):
        if isinstance(self.criterion, (nn.BCEWithLogitsLoss, nn.BCELoss)):
            y = y.double()
        val = self.criterion(Variable(x), Variable(y)).data
        if torch.is_tensor(val) and val.dim() == 0:
            return super(LossMeter, self).measure(val.item())
        else:
            return super(LossMeter, self).measure(val[0])
