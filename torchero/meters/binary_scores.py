import torch
from torch import nn
from .batch import BatchMeter
from .base import BaseMeter

class BinaryAccuracy(BatchMeter):
    """ Meter for accuracy on binary targets (assuming normalized inputs)
    """
    INVALID_DIMENSION_MESSAGE = 'Expected both tensors have same dimension'
    INVALID_INPUT_TYPE_MESSAGE = 'Expected Tensors as inputs'
    INVALID_TENSOR_CONTENT_MESSAGE = 'Expected binary target tensors (1 or 0 in each component)'

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
        return (predictions == target).float()

    def check_tensors(self, a, b):
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if not (a.shape == b.shape):
            raise ValueError(self.INVALID_DIMENSION_MESSAGE)

        if not ((b == 0) | (b == 1)).all():
            raise ValueError(self.INVALID_TENSOR_CONTENT_MESSAGE)

class BinaryWithLogitsAccuracy(BinaryAccuracy):
    """ Binary accuracy meter with an integrated activation function
    """
    def __init__(self, aggregator=None, threshold=0.5, activation=None):
        super(BinaryWithLogitsAccuracy, self).__init__(threshold=threshold,
                                                       aggregator=aggregator)
        self.activation = activation
        if self.activation is None:
            self.activation = nn.Sigmoid()

    def _get_result(self, output, target):
        return super(BinaryWithLogitsAccuracy, self)._get_result(self.activation(output),
                                                                 target)

class TPMeter(BaseMeter):
    """ Meter to calculate true positives, true negatives, false positives,
    false negatives
    """
    INVALID_DIMENSION_MESSAGE = 'Expected both tensors have same dimension'
    INVALID_INPUT_TYPE_MESSAGE = 'Expected Tensors as inputs'
    INVALID_TENSOR_CONTENT_MESSAGE = 'Expected binary target tensors (1 or 0 in each component)'

    def __init__(self, threshold=0.5, with_logits=False, activation=None):
        super(TPMeter, self).__init__()
        self.threshold = threshold
        if with_logits and activation is None:
            self.activation = nn.Sigmoid()
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
        self.tp += (predictions & target).sum().item()
        self.fp += (predictions & (target^1)).sum().item()
        self.fn += ((predictions^1) & target).sum().item()
        self.tn += ((predictions^1) & (target^1)).sum().item()

    def value(self):
        return (self.tp, self.tn, self.fp, self.fn)

class Recall(TPMeter):
    """ Meter to calculate the recall score where

    recall = tp / (tp + fn)

    The best value is 1 and the worst value is 0.
    """
    def value(self):
        if (self.tp + self.fn == 0):
            return 0
        else:
            return self.tp/(self.tp + self.fn)

class Precision(TPMeter):
    """ Meter to calculate the precision score where

    precision = tp / (tp + fp)

    The best value is 1 and the worst value is 0.
    """
    def value(self):
        if (self.tp + self.fp == 0):
            return 0
        else:
            return self.tp/(self.tp + self.fp)

class Specificity(TPMeter):
    """ Meter to calculate the specificity score where

    specificity = tn / (tn + fp)

    The best value is 0 and the worst value is 1.
    """
    def value(self):
        return self.tn/(self.tn + self.fp)

class NPV(TPMeter):
    """ Meter to calculate the negative predictive value score (npv) where

    npv = tn / (tn + fp)

    The best value is 0 and the worst value is 1.
    """
    def value(self):
        return self.tn/(self.tn + self.fn)

class FBetaScore(TPMeter):
    """ Meter to calculate the f-beta score where

    f(beta) = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall')

    The best value is 1 and the worst value is 0.
    """
    def __init__(self, beta, threshold=0.5):
        super(FBetaScore, self).__init__(threshold=threshold)
        self.beta = beta

    def value(self):
        recall = self.tp/(self.tp + self.fn)
        precision = self.tp/(self.tp + self.fp)
        return (1 + self.beta**2) * precision * recall / ((self.beta **2) * precision + recall)

class F1Score(FBetaScore):
    """ Meter to calculate the f1 score where

    f1 = 2 * (precision * recall) / (precision + recall)

    The best value is 1 and the worst value is 0.
    """
    def __init__(self, threshold=0.5):
        super(F1Score, self).__init__(threshold=threshold, beta=1)

class F2Score(FBetaScore):
    """ Meter to calculate the f1 score where

    f2 = 5 * (precision * recall) / (4 * precision + recall)

    The best value is 1 and the worst value is 0.
    """
    def __init__(self, threshold=0.5):
        super(F2Score, self).__init__(threshold=threshold, beta=2)

class FHalfScore(FBetaScore):
    """ Meter to calculate the f0.5 score where

    f0.5 = 1.25 * (precision * recall) / (0.25 * precision + recall)

    The best value is 1 and the worst value is 0.
    """
    def __init__(self, threshold=0.5):
        super(FHalfScore, self).__init__(threshold=threshold, beta=0.5)
