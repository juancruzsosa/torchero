import torch
from torch import nn

from torchero.meters.base import BaseMeter
from torchero.meters.batch import BatchMeter


class BinaryAccuracy(BatchMeter):
    """ Meter for accuracy on binary targets (assuming inputs in range (0,1))
    """
    name = "acc"
    DEFAULT_MODE = 'max'
    INVALID_DIMENSION_MESSAGE = (
        'Expected both tensors have same dimension'
    )
    INVALID_INPUT_TYPE_MESSAGE = (
        'Expected Tensors as inputs'
    )
    INVALID_TENSOR_CONTENT_MESSAGE = (
        'Expected binary target tensors (1 or 0 in each component)'
    )

    def __init__(self, threshold=0.5, aggregator=None):
        """ Constructor

        Arguments:
            threshold (float): Positive/Negative class separation threshold
        """
        super(BinaryAccuracy, self).__init__(aggregator=aggregator)
        self.threshold = threshold

    def _get_result(self, output, target):
        predictions = output >= self.threshold

        if predictions.is_cuda:
            predictions = predictions.type(torch.cuda.LongTensor)
        else:
            predictions = predictions.type(torch.LongTensor)

        if target.is_cuda:
            target = target.type(torch.cuda.LongTensor)
        else:
            target = target.type(torch.LongTensor)
        return (predictions == target).float().cpu()

    def check_tensors(self, a, b):
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if not (a.shape == b.shape):
            raise ValueError(self.INVALID_DIMENSION_MESSAGE)

        if not ((b == 0) | (b == 1)).all():
            raise ValueError(self.INVALID_TENSOR_CONTENT_MESSAGE)

    def __repr__(self):
        return "{cls}(threshold={th}, aggregator={agg})".format(
            cls=self.__class__.__name__,
            th=repr(self.threshold),
            agg=repr(self.aggregator)
        )

class BinaryWithLogitsAccuracy(BinaryAccuracy):
    """ Binary accuracy meter with an integrated activation function
    """
    DEFAULT_MODE = 'max'
    def __init__(self, aggregator=None, threshold=0.5, activation=None):
        super(BinaryWithLogitsAccuracy, self).__init__(threshold=threshold,
                                                       aggregator=aggregator)
        self.activation = activation
        if self.activation is None:
            self.activation = nn.Sigmoid()

    def _get_result(self, output, target):
        output = self.activation(output)
        return super(BinaryWithLogitsAccuracy, self)._get_result(output,
                                                                 target)
    def __repr__(self):
        return "{cls}(threshold={th}, aggregator={agg}, activation={act})".format(
            cls=self.__class__.__name__,
            th=repr(self.threshold),
            agg=repr(self.aggregator),
            act=repr(self.activation)
        )

class TPMeter(BaseMeter):
    """ Meter to calculate true positives, true negatives, false positives,
    false negatives
    """
    INVALID_DIMENSION_MESSAGE = (
        'Expected both tensors have same dimension'
    )
    INVALID_INPUT_TYPE_MESSAGE = (
        'Expected Tensors as inputs'
    )
    INVALID_TENSOR_CONTENT_MESSAGE = (
        'Expected binary target tensors (1 or 0 in each component)'
    )

    def __init__(self, threshold=0.5, with_logits=False, activation=None, agg='micro'):
        """ Constructor

        Arguments:
            threshold (float): Activation threshold used
            with_logits (bool): Set this as `True` if your model does not
                contain sigmoid as activation in the final layer (preferable)
                or 'False' otherwise
            activation (nn.Sigmoid): Use a custom activation instead of
                sigmoid. Default: None (use nn.Sigmoid if with_logits=True)
        """
        super(TPMeter, self).__init__()
        self.threshold = threshold
        if with_logits and activation is None:
            activation = nn.Sigmoid()
        if agg not in ('micro', 'macro', 'weighted'):
            raise ValueError("agg parameter must be micro, or macro")
        self.agg = agg
        self.with_logits = with_logits
        self.activation = activation
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def check_tensors(self, a, b):
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if not (a.shape == b.shape):
            raise ValueError(self.INVALID_DIMENSION_MESSAGE)

        if not ((b == 0) | (b == 1)).all():
            raise ValueError(self.INVALID_TENSOR_CONTENT_MESSAGE)

    def measure(self, output, target):
        self.check_tensors(output, target)
        target = target.long()
        if self.activation is not None:
            output = self.activation(output)
        predictions = (output >= self.threshold).long()
        self.tp += (predictions & target).sum(dim=0).cpu()
        self.fp += (predictions & (target ^ 1)).sum(dim=0).cpu()
        self.fn += ((predictions ^ 1) & target).sum(dim=0).cpu()
        self.tn += ((predictions ^ 1) & (target ^ 1)).sum(dim=0).cpu()

    def support(self):
        return self.tp + self.fn

    def aggregate(self, f, agg=None):
        agg = agg or self.agg
        tp, tn, fp, fn = self.tp, self.tn, self.fp, self.fn
        if agg == 'micro':
            return f(tp.sum(), tn.sum(), fp.sum(), fn.sum()).item()
        elif agg == 'macro':
            return f(tp, tn, fp, fn).mean().item()
        else: # agg == 'weighted'
            support = self.support()
            return ((f(tp, tn, fp, fn) * support).sum()/support.sum()).item()

    @staticmethod
    def _recall(tp, tn, fp, fn):
        return torch.where(tp+fn==0, torch.zeros_like(tp, dtype=torch.float), tp.float()/(tp + fn).float())

    @staticmethod
    def _precision(tp, tn, fp, fn):
        return torch.where(tp+fp==0, torch.zeros_like(tp, dtype=torch.float), tp.float()/(tp + fp).float())

    @staticmethod
    def _specificity(tp, tn, fp, fn):
        return torch.where(tn+fn==0, torch.zeros_like(tp, dtype=torch.float), tn.float()/(tn + fp).float())

    @staticmethod
    def _npv(tp, tn, fp, fn):
        return torch.where(tn+fn==0, torch.zeros_like(tp, dtype=torch.float), tn.float()/(tn + fn).float())

    def _gen_fbeta(self, beta=1):
        def fbeta(tp, tn, fp, fn):
            r = self._recall(tp, tn, fp, fn)
            p = self._precision(tp, tn, fp, fn)
            num = (1 + beta**2) * p * r
            den = (beta ** 2) * p + r
            return torch.where(den == 0, torch.zeros_like(r, dtype=torch.float), num / den)
        return fbeta

    @property
    def recall(self):
        return self.aggregate(self._recall)

    @property
    def precision(self):
        return self.aggregate(self._precision)

    @property
    def specificity(self):
        return self.aggregate(self._specificity)

    @property
    def npv(self):
        return self.aggregate(self._npv)

    def f_beta(self, beta):
        return self.aggregate(self._gen_fbeta(beta))

    def value(self):
        return (self.tp, self.tn, self.fp, self.fn)

    def __repr__(self):
        return "{cls}(threshold={th}, with_logits={wl}, activation={act}, agg={agg})".format(
            cls=self.__class__.__name__,
            th=repr(self.threshold),
            wl=repr(self.with_logits),
            act=repr(self.activation),
            agg=repr(self.agg),
        )

class BinaryClassificationReport(TPMeter):
    """ Classification report meter for multi-label classification
    """
    def __init__(self, threshold=0.5, with_logits=False, activation=None, names=None):
        super(BinaryClassificationReport, self).__init__(threshold=threshold,
                                                         with_logits=with_logits,
                                                         activation=activation)
        self.names = names

    def value(self):
        result = {}
        original_agg = self.agg
        single_label = self.tp.ndim == 0
        if single_label:
            self.tp = self.tp.unsqueeze(0)
            self.tn = self.tn.unsqueeze(0)
            self.fp = self.fp.unsqueeze(0)
            self.fn = self.fn.unsqueeze(0)
        if self.names is not None:
            if len(self.names) != len(self.tp):
                raise ValueError('Names and number of labels length mismatch!')
            names = self.names
        else:
            names = range(len(self.tp))
        for i, name in enumerate(names):
            metrics = {
                'precision': self._precision(self.tp[i], self.tn[i], self.fp[i], self.fn[i]).item(),
                'recall':    self._recall(self.tp[i], self.tn[i], self.fp[i], self.fn[i]).item(),
                'f1-score':  self._gen_fbeta(1)(self.tp[i], self.tn[i], self.fp[i], self.fn[i]).item(),
                'support':  (self.tp[i] + self.fn[i]).item()
            }
            result[name] = metrics
        if not single_label:
            for agg in ['micro', 'macro', 'weighted']:
                self.agg = agg
                metrics = {'precision': self.precision,
                           'recall': self.recall,
                           'f1-score': self.f_beta(1),
                           'support': self.support().sum().item()}
                result[agg] = metrics
        return result


class Recall(TPMeter):
    """ Meter to calculate the recall score where

    recall = tp / (tp + fn)

    The best value is 1 and the worst value is 0.
    """
    DEFAULT_MODE = 'max'

    def value(self):
        return self.recall


class Precision(TPMeter):
    """ Meter to calculate the precision score where

    precision = tp / (tp + fp)

    The best value is 1 and the worst value is 0.
    """

    DEFAULT_MODE = 'max'
    def value(self):
        return self.precision


class Specificity(TPMeter):
    """ Meter to calculate the specificity score where

    specificity = tn / (tn + fp)

    The best value is 0 and the worst value is 1.
    """
    DEFAULT_MODE = 'min'

    def value(self):
        return self.specificity


class NPV(TPMeter):
    """ Meter to calculate the negative predictive value score (npv) where

    npv = tn / (tn + fp)

    The best value is 0 and the worst value is 1.
    """
    DEFAULT_MODE = 'min'

    def value(self):
        return self.npv


class FBetaScore(TPMeter):
    """ Meter to calculate the f-beta score where

    f(beta) = (1 + beta**2) * (precision * recall) /
                (beta**2 * precision + recall')

    The best value is 1 and the worst value is 0.
    """
    DEFAULT_MODE = 'max'

    def __init__(self, beta, threshold=0.5, with_logits=False):
        """ Constructor

        Arguments:
            threshold (float): Activation threshold used
            beta (float): Beta
            with_logits (bool): Set this as `True` if your model does not
                contain sigmoid as activation in the final layer (preferable)
                or 'False' otherwise
            activation (nn.Sigmoid): Use a custom activation instead of
                sigmoid. Default: None (use nn.Sigmoid if with_logits=True)
        """
        super(FBetaScore, self).__init__(threshold=threshold,
                                         with_logits=with_logits)
        self.beta = beta

    def value(self):
        return self.f_beta(self.beta)


class F1Score(FBetaScore):
    """ Meter to calculate the f1 score where

    f1 = 2 * (precision * recall) / (precision + recall)

    The best value is 1 and the worst value is 0.
    """
    name = "f1"

    def __init__(self, threshold=0.5, with_logits=False):
        super(F1Score, self).__init__(threshold=threshold,
                                      beta=1,
                                      with_logits=with_logits)


class F2Score(FBetaScore):
    """ Meter to calculate the f1 score where

    f2 = 5 * (precision * recall) / (4 * precision + recall)

    The best value is 1 and the worst value is 0.
    """
    name = "f2"

    def __init__(self, threshold=0.5, with_logits=False):
        super(F2Score, self).__init__(threshold=threshold,
                                      beta=2,
                                      with_logits=with_logits)


class FHalfScore(FBetaScore):
    """ Meter to calculate the f0.5 score where

    f0.5 = 1.25 * (precision * recall) / (0.25 * precision + recall)

    The best value is 1 and the worst value is 0.
    """
    name = "fh"

    def __init__(self, threshold=0.5, with_logits=False):
        super(FHalfScore, self).__init__(threshold=threshold,
                                         beta=0.5,
                                         with_logits=with_logits)
