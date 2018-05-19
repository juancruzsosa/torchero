from torch.autograd import Variable
from .averager import Averager

class LossMeter(Averager):
    def __init__(self, criterion):
        super(Averager, self).__init__()
        self.criterion = criterion

    def measure(self, x, y):
        return super(LossMeter, self).measure(self.criterion(Variable(x), Variable(y)).data[0])
