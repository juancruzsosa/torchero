import torch
from torch.autograd import Variable
from .averager import Averager

class LossMeter(Averager):
    def __init__(self, criterion):
        super(Averager, self).__init__()
        self.criterion = criterion

    def measure(self, x, y):
        val = self.criterion(Variable(x), Variable(y)).data
        if torch.is_tensor(val) and val.dim() == 0:
            return super(LossMeter, self).measure(val.item())
        else:
            return super(LossMeter, self).measure(val[0])
