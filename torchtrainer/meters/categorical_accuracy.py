import torch
from torch import nn
from .batch import BatchMeter

class _CategoricalAccuracy(BatchMeter):
    DEFAULT_MODE = 'max'
    INVALID_BATCH_DIMENSION_MESSAGE = ('Expected both tensors have at less two '
                                       'dimension and same shape')
    INVALID_INPUT_TYPE_MESSAGE = 'Expected types (Tensor, LongTensor) as inputs'

    def check_tensors(self, a, b):
        if not torch.is_tensor(a):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if (not isinstance(b, torch.LongTensor) and
            not isinstance(b, torch.cuda.LongTensor)):
            raise TypeError(self.INVALID_INPUT_TYPE_MESSAGE)

        if len(a.size()) != 2 or len(b.size()) != 1 or len(b) != a.size()[0]:
            raise ValueError(self.INVALID_BATCH_DIMENSION_MESSAGE)

class CategoricalAccuracy(_CategoricalAccuracy):
    """ Meter for accuracy categorical on categorical targets
    """
    def __init__(self, k=1, aggregator=None):
        super(CategoricalAccuracy, self).__init__(aggregator=aggregator)
        self.k = k

    def _get_result(self, a, b):
        return torch.sum((a.topk(k=self.k, dim=1)[1] == b.unsqueeze(-1)).float(), dim=1)

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
