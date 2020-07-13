import copy
from abc import ABCMeta, abstractmethod

import torch


class BaseMeter(object, metaclass=ABCMeta):
    """ Interface for all meters.
    All meters should subclass this class
    """
    @abstractmethod
    def measure(self, *batchs):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def value(self):
        pass

    def clone(self):
        return copy.deepcopy(self)

    def eval_model_on_dl(self, model, dl):
        is_cuda = next(model.parameters()).is_cuda
        self.reset()
        with torch.no_grad():
            model.eval()
            for x, y_hat in dl:
                if is_cuda:
                    x = x.cuda()
                    y_hat = y_hat.cuda()

                self.measure(model(x.cuda()), y_hat.cuda())
        return self.value()
