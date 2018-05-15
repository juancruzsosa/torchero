import os
import unittest
import torch
from torch import nn
from torch.optim import SGD
from torchtrainer.base import BatchTrainer
from torchtrainer.callbacks import Callback, History, CSVLogger, ModelCheckpoint, MeterNotFound
from torchtrainer import meters
from torchtrainer.meters import Averager, MSE
from torch.utils.data import DataLoader, TensorDataset
from torchtrainer.utils.data import CrossFoldValidation
from torchtrainer.base import ValidationGranularity

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.train(mode=True)
        self.is_cuda = False

    def cuda(self):
        self.is_cuda = True

    def cpu(self):
        self.is_cuda = False

    def forward(self, x):
        return x

class TestTrainer(BatchTrainer):
    def __init__(self, *args, **kwargs):
        self.update_batch_fn = None
        self.valid_batch_fn = None

        try:
            self.update_batch_fn = kwargs.pop('update_batch_fn')
        except KeyError:
            pass

        try:
            self.valid_batch_fn = kwargs.pop('valid_batch_fn')
        except KeyError:
            pass

        if 'validation_granularity' not in kwargs:
            kwargs['validation_granularity'] = ValidationGranularity.AT_LOG


        super(TestTrainer, self).__init__(*args, **kwargs)

    def update_batch(self, *args, **kwargs):
        if self.update_batch_fn:
            self.update_batch_fn(self, *args, **kwargs)

    def validate_batch(self, *args, **kwargs):
        if self.valid_batch_fn:
            self.valid_batch_fn(self, *args, **kwargs)

def requires_cuda(f):
    def closure(*args, **kwargs):
        if not torch.cuda.is_available():
            print("Skipping `{}´ test cause use CUDA but CUDA isn't available !!".format(f.__name__), file=sys.stderr)
            return
        return f(*args, **kwargs)
    return closure

