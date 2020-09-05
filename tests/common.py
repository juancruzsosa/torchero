import sys
import os
import unittest
import torch
from torch import nn
from torch.optim import SGD
from torchero.base import BatchTrainer, BatchValidator
from torchero.callbacks import Callback, History, CSVLogger, ModelCheckpoint, MeterNotFound, EarlyStopping, ProgbarLogger
from torchero import meters
from torchero.meters import Averager, MSE, RMSE, CategoricalAccuracy
from torch.utils.data import DataLoader, TensorDataset
from torchero.utils.data import CrossFoldValidation
from torchero.base import ValidationGranularity

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.dummy = nn.Linear(1, 1)
        self.train(mode=True)

    @property
    def is_cuda(self):
        return next(iter(self.parameters())).device.type == 'cuda'

    def forward(self, x):
        return x

class BinaryNetwork(nn.Module):
    def __init__(self):
        super(BinaryNetwork, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1)
        return x

class TestValidator(BatchValidator):
    def __init__(self, model, meters, trainer=None):
        super(TestValidator, self).__init__(model, meters)
        self.trainer = trainer

    def validate_batch(self, *args, **kwargs):
        if self.trainer.valid_batch_fn:
            self.trainer.valid_batch_fn(self, *args, **kwargs)

class TestTrainer(BatchTrainer):
    def create_validator(self, meters):
        return TestValidator(self.model, meters, self)

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


def requires_cuda(f):
    def closure(*args, **kwargs):
        if not torch.cuda.is_available():
            print("Skipping `{}´ test cause use CUDA but CUDA isn't available !!".format(f.__name__), file=sys.stderr)
            return
        return f(*args, **kwargs)
    return closure

