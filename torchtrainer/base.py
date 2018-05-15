import torch
from torch.autograd import Variable
from abc import ABCMeta, abstractmethod
from .callbacks import Callback, CallbackContainer
from .meters import ZeroMeasurementsError
from enum import Enum
from itertools import chain

class ValidationGranularity(Enum):
    AT_LOG='log'
    AT_EPOCH='epoch'

class _OnLogValidScheduler(Callback):
    def on_log(self):
        self.trainer._validate()

class _OnEpochValidScheduler(Callback):
    def on_log(self):
        if self.trainer.step == self.trainer.total_steps-1:
            self.trainer._validate()

class BatchTrainer(object, metaclass=ABCMeta):
    """ Abstract trainer for all trainer classes that works with batched inputs.
        All those trainers should subclass this class
    """

    INVALID_EPOCH_MESSAGE=('Expected epoch to be a non-negative integer, '
                           'got: {epochs}')
    INVALID_LOGGING_FRECUENCY_MESSAGE=('Expected loggin frecuency to be a '
                                       'non-negative integer, '
                                       'got: {logging_frecuency}')
    INVALID_VALIDATION_GRANULARITY_MESSAGE=('Expected logging frecuency to be '
                                            'one of '
                                            'ValidationGranularity.AT_LOG\' or '
                                            'ValidationGranularity.AT_EPOCH\' '
                                            'got: {mode}')

    SCHED_BY_GRANULARITY = {ValidationGranularity.AT_EPOCH : _OnEpochValidScheduler,
                            ValidationGranularity.AT_LOG: _OnLogValidScheduler}

    def __init__(self,
                 model,
                 callbacks=[],
                 train_meters={}, val_meters={},
                 logging_frecuency=1,
                 validation_granularity=ValidationGranularity.AT_LOG):
        """ Constructor

        Args:
            model (:class:`torch.nn.Module`):
                Module to train
            callbacks (:class:`torchtrainer.callbacks.Callback`):
                Pluggable callbacks for epoch/batch events.
            logging_frecuency (int):
                Frecuency of log to monitor train/validation
        """
        if logging_frecuency < 0:
            raise Exception(self.INVALID_LOGGING_FRECUENCY_MESSAGE.format(logging_frecuency=logging_frecuency))

        if validation_granularity not in ValidationGranularity:
            raise Exception(self.INVALID_VALIDATION_GRANULARITY_MESSAGE.format(mode=validation_granularity))

        valid_sched = self.SCHED_BY_GRANULARITY[validation_granularity]()

        self.logging_frecuency = logging_frecuency

        self.model = model
        self._epochs_trained = 0
        self._use_cuda = False
        self._train_metrics = {}
        self._val_metrics = {}
        self.train_meters = train_meters
        self.val_meters = val_meters

        self._callbacks = CallbackContainer()
        self._callbacks.accept(self)
        self._callbacks.add(valid_sched)
        for callback in callbacks:
            self._callbacks.add(callback)

    def cuda(self):
        """ Turn model to cuda
        """
        self._use_cuda = True
        self.model.cuda()

    def cpu(self):
        """ Turn model to cpu
        """
        self._use_cuda = False
        self.model.cpu()

    def _to_variable(self, x):
        if self._use_cuda:
            x = x.cuda()
        return Variable(x)

    def meters_names(self):
        """ Returns the meters names
        """
        return sorted(chain(self.train_meters.keys(),
                            self.val_meters.keys()))

    @property
    def metrics(self):
        """ Last statistic recopiled from meters

        Returns
            dict: Dictionary of metric name and value, one for each
            `meters` that made at least one measure
        """
        return {**self._train_metrics, **self._val_metrics}

    def _compile_train_metrics(self):
        self._train_metrics = {}

        for metric_name, meter in self.train_meters.items():
            try:
                self._train_metrics[metric_name] = meter.value()
            except ZeroMeasurementsError:
                continue

    def _compile_val_metrics(self):
        self._val_metrics = {}

        for metric_name, meter in self.val_meters.items():
            try:
                self._val_metrics[metric_name] = meter.value()
            except ZeroMeasurementsError:
                continue

    @property
    def epochs_trained(self):
        """ Total number of epochs epochs_trained

        Returns:
            int: number of epochs
        """
        return self._epochs_trained

    @epochs_trained.setter
    def epochs_trained(self, value):
        if value < 0:
            raise AttributeError('can\'t set epochs_trained'
                                 'to a value less than zero')

    @abstractmethod
    def update_batch(self, *args, **kwargs):
        """ Abstract method for update model parameters given a batch

        Args:
            *args (variable length arguments of :class:`torch.autograd.Variable`
                   of Tensors or cuda Tensors):
                Unamed batch parameters
            **kwargs (variable length keyword arguments of
                      :class:`torch.autograd.Variable` of
                      Tensors or cuda Tensors):
                Named batch parameters
        """

        pass

    def reset_meters(self):
        for meter in chain(self.train_meters.values(), self.val_meters.values()):
            meter.reset()

    def log(self):
        self._callbacks.on_log()

    def log_started(self):
        return self.logging_frecuency > 0 and self.step % self.logging_frecuency == 0

    def _train_epoch(self, train_dataloader, valid_dataloader=None):
        for self.step, batch in enumerate(train_dataloader):
            if self.log_started():
                self.reset_meters()

            # convert to 1-d tuple if batch was a tensor instead of a tuple
            if torch.is_tensor(batch):
                batch = (batch, )
            batch = list(map(self._to_variable, batch))
            self.update_batch(*batch)

            if self._is_time_to_log():
                self._compile_train_metrics()
                self.log()

        self._epochs_trained += 1

    def train(self, dataloader, valid_dataloader=None, epochs=1):
        """ Train the model

        Args:
            dataloader (:class:`torch.utils.DataLoader`):
                Train data loader
            valid_dataloader (:class:`torch.utils.DataLoader`):
                Validation data loader
            epochs (int):
                Number of epochs to train
        """
        if epochs < 0:
            raise Exception(self.INVALID_EPOCH_MESSAGE.format(epochs=epochs))

        self.total_epochs = epochs
        self.total_steps = len(dataloader)
        self.valid_dataloader = valid_dataloader

        self._callbacks.on_train_begin()

        # Turn model to training mode
        self.model.train(mode=True)

        for self.epoch in range(self.total_epochs):
            self._callbacks.on_epoch_begin()
            self._train_epoch(dataloader, valid_dataloader)
            self._callbacks.on_epoch_end()

        self._callbacks.on_train_end()

        del self.valid_dataloader

        # Turn model to evaluation mode
        self.model.train(mode=False)

    def _is_time_to_log(self):
        log_frec = self.logging_frecuency
        return log_frec > 0 and ((self.total_steps % log_frec != 0 and
                                  self.step == self.total_steps - 1)
               or self.step % log_frec == log_frec - 1)

    @abstractmethod
    def validate_batch(self, *arg, **kwargs):
        """ Abstract method for validate model per batch

        Args:
            *args (variable length arguments of :class:`torch.autograd.Variable`
                   of Tensors or cuda Tensors):
                Unamed batch parameters
            **kwargs (variable length keyword arguments of
                      :class:`torch.autograd.Variable` of
                      Tensors or cuda Tensors):
                Named batch parameters
        """
        pass

    def _validate(self):
        if self.valid_dataloader is None:
            return

        self.model.train(mode=False)
        for valid_batch in self.valid_dataloader:
            if isinstance(valid_batch, torch.Tensor):
                valid_batch = (valid_batch, )
            valid_batch = list(map(self._to_variable, valid_batch))
            self.validate_batch(*valid_batch)
        self.model.train(mode=True)
        self._compile_val_metrics()
