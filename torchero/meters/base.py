import copy
from abc import ABCMeta, abstractmethod

import torch


class BaseMeter(object, metaclass=ABCMeta):
    """ Interface for all meters.
    All meters should subclass this class
    """
    @abstractmethod
    def measure(self, *batchs):
        """ Partial calculation of the metric for the given batches
        """
        pass

    @abstractmethod
    def reset(self):
        """ Resets the metric value
        """
        pass

    @abstractmethod
    def value(self):
        """ Returns the metric final value
        """
        pass

    def clone(self):
        """ Create a new instance copied from this instance
        """
        return copy.deepcopy(self)

    def eval_model_on_dl(self, model, dl):
        """ Evaluates the model on a given DataLoader

        Arguments:
            model (nn.Module): Module to run the metric against
            dl (DataLoader): Input DataLoader
        """
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

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)
