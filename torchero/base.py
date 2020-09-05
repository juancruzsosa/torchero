from abc import ABCMeta, abstractmethod
from enum import Enum
from itertools import chain
from collections.abc import MutableMapping

import torch
from torch.autograd import Variable

import torchero.hparams
from torchero.callbacks import Callback, CallbackContainer, History
from torchero.meters import ZeroMeasurementsError
from torchero.utils.defaults import parse_meters
from torchero.utils.mixins import DeviceMixin


class ValidationGranularity(Enum):
    AT_LOG = 'log'
    AT_EPOCH = 'epoch'


class _OnLogValidScheduler(Callback):
    def on_log(self):
        self.trainer._validate()


class _OnEpochValidScheduler(Callback):
    def on_log(self):
        if self.trainer.step == self.trainer.total_steps-1:
            self.trainer._validate()

class ParamsDict(MutableMapping):
    """ Container for hyperparameters dictionary
    """
    def __init__(self, mapping=(), **kwargs):
        """ Constructor

        Arguments:
            mapping (dict or list of key-val tuples):
                Build from existent mapping
        """
        self._hparams = dict(mapping, **kwargs)

    def __getitem__(self, key):
        """ Retrives the value of the given hyperparameter

        Arguments:
            key (str): Hyperparameter name

        Returns:
            The value of the hyperparameter ``key``
            if it exists, otherwise raises a KeyError
        """
        return self._hparams[key].value

    def __delitem__(self, key):
        """ Removes a hyperparameter from the dict

        Arguments:
            key (str): Hyperparameter name to delete
        """
        del self._hparams[key]

    def __setitem__(self, key, value):
        """ Sets the value of a hyperparameter to a new value

        Arguments:
            key (str): Hyperparameter name
            value: New value for the hyperparameter
        """
        self._hparams[key].value = value

    def __iter__(self):
        """ Same as keys() method
        """
        return self.keys()

    def keys(self):
        """ Returns an iterator for the hyperparameters names
        """
        return iter(self._hparams.keys())

    def values(self):
        """ Returns an iterator for the hyperparameters values
        """
        yield from map(lambda x: x.value, self._hparams.values())

    def items(self):
        """ Returns an iterator of (hyp. name, hyp. value) items
        """
        yield from map(lambda x: (x[0], x[1].value), self._hparams.items())

    def __len__(self):
        """ Returns the numbers of hyperparameters
        """
        return len(self._hparams)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__,
                               ", ".join(map(lambda x: "{}={}".format(x[0], repr(x[1])), self)))



class BatchValidator(DeviceMixin, metaclass=ABCMeta):
    """ Abstract class for all validation classes that works with batched inputs.
        All those validators should subclass this class
    """
    METER_ALREADY_EXISTS_MESSAGE = 'Meter {name} already exists as train meter'

    def __init__(self, model, meters):
        super(BatchValidator, self).__init__()
        self.model = model
        meters = parse_meters(meters)
        self._meters = meters
        self._metrics = {}

    def _prepare_tensor(self, x):
        if torch.is_tensor(x):
            return Variable(self._convert_tensor(x))
        else:
            return x

    @abstractmethod
    def validate_batch(self, *arg, **kwargs):
        """ Abstract method for validate model per batch

        Args:
            *args (variable length arguments
                   of :class:`torch.autograd.Variable`
                   of Tensors or cuda Tensors):
                Unamed batch parameters
            **kwargs (variable length keyword arguments of
                      :class:`torch.autograd.Variable` of
                      Tensors or cuda Tensors):
                Named batch parameters
        """
        pass

    def meters_names(self):
        return self._meters.keys()

    @property
    def meters(self):
        return self._meters

    def _compile_metrics(self):
        for metric_name, meter in self._meters.items():
            try:
                value = meter.value()
                self._metrics[metric_name] = value
            except ZeroMeasurementsError:
                continue

    def _reset_meters(self):
        self._metrics = {}

        for meter in self._meters.values():
            meter.reset()

    def validate(self, valid_dataloader):
        self._reset_meters()

        if not valid_dataloader:
            return self._metrics

        self.model.train(mode=False)
        with torch.no_grad():
            for batch in valid_dataloader:
                if isinstance(batch, torch.Tensor):
                    batch = (batch, )
                batch = list(map(self._prepare_tensor, batch))
                self.validate_batch(*batch)
        self.model.train(mode=True)

        self._compile_metrics()
        return self._metrics

    def add_named_meter(self, name, meter):
        if name in self._meters:
            raise Exception(self.METER_ALREADY_EXISTS_MESSAGE
                                .format(name=name))

        self._meters[name] = meter


class BatchTrainer(DeviceMixin, metaclass=ABCMeta):
    """ Abstract trainer for all trainer classes that works with batched inputs.
        All those trainers should subclass this class
    """

    INVALID_EPOCH_MESSAGE = (
        'Expected epoch to be a non-negative integer, got: {epochs}'
    )
    INVALID_LOGGING_FRECUENCY_MESSAGE = (
        'Expected loggin frecuency to be a '
        'non-negative integer, got: {logging_frecuency}'
    )
    INVALID_VALIDATION_GRANULARITY_MESSAGE = (
        'Expected logging frecuency '
        'to be one of ValidationGranularity.AT_LOG\' '
        'or ValidationGranularity.AT_EPOCH\' got: {mode}'
    )
    METER_ALREADY_EXISTS_MESSAGE = (
        'Meter {name} already exists as train meter'
    )
    SCHED_BY_GRANULARITY = {
        ValidationGranularity.AT_EPOCH: _OnEpochValidScheduler,
        ValidationGranularity.AT_LOG: _OnLogValidScheduler
    }

    @staticmethod
    def prepend_name_dict(prefix, d):
        return {prefix + name: value for name, value in d.items()}

    @abstractmethod
    def create_validator(self, metrics):
        # return BatchValidator(self.model, self.val_meters)
        pass

    def __init__(self,
                 model,
                 callbacks=[],
                 train_meters={}, val_meters={},
                 hparams={},
                 logging_frecuency=1,
                 prefixes=('', ''),
                 validation_granularity=ValidationGranularity.AT_EPOCH):
        """ Constructor

        Args:
            model (:class:`torch.nn.Module`):
                Module to train
            callbacks (:class:`torchero.callbacks.Callback`):
                Pluggable callbacks for epoch/batch events.
            train_meters (list or dict of :class: `torchero.meters.Meter'):
                Training meters
            val_meters (list dict of :class: `torchero.meters.Meter'):
                Validation meters
            hparams (dict of hyperparameters):
                Dictionary <name,hparam> of hyperparameters. Each value
                can be a fixed value, a lambda function with only parameter to pass the trainer,
                or a instance of `torchero.hparams.OptimP`
            logging_frecuency (int):
                Frecuency of log to monitor train/validation
            prefixes (tuple, list):
                Prefixes of train and val metrics
            validation_granularity (ValidationGranularity):
                Change validation criterion (after every log vs after every
                epoch)
        """
        if logging_frecuency < 0:
            raise Exception(self.INVALID_LOGGING_FRECUENCY_MESSAGE
                                .format(logging_frecuency=logging_frecuency))

        if not isinstance(validation_granularity, ValidationGranularity) \
                or validation_granularity not in ValidationGranularity:
            raise Exception(self.INVALID_VALIDATION_GRANULARITY_MESSAGE
                                .format(mode=validation_granularity))

        super(BatchTrainer, self).__init__()

        valid_sched = self.SCHED_BY_GRANULARITY[validation_granularity]()

        self.logging_frecuency = logging_frecuency

        self.model = model
        self._epochs_trained = 0
        self._steps_trained = 0
        self._train_metrics = {}
        self._val_metrics = {}
        self._prefixes = prefixes

        train_meters = parse_meters(train_meters)

        if val_meters is None:
            val_meters = {name: meter.clone()
                          for name, meter in train_meters.items()}
        else:
            val_meters = parse_meters(val_meters)

        self.train_meters = train_meters
        self._hparams = self._create_hparams(dict(hparams))
        self.validator = self.create_validator(val_meters)

        self._raised_stop_training = False

        self._history_callback = History()

        self._callbacks = CallbackContainer()
        self._callbacks.accept(self)

        self._callbacks.add(valid_sched)
        self._callbacks.add(self._history_callback)

        for callback in callbacks:
            self._callbacks.add(callback)

    @property
    def history(self):
        return self._history_callback.registry

    def to(self, device):
        super(BatchTrainer, self).to(device)
        self.model.to(self._device)
        self.validator.to(self._device)

    def meters_names(self):
        """ Returns the meters names
        """
        return sorted(self.meters.keys())

    @property
    def meters(self):
        return {**self.prepend_name_dict(self._prefixes[0], self.train_meters),
                **self.prepend_name_dict(self._prefixes[1], self.validator.meters)}

    @property
    def metrics(self):
        """ Last statistic recopiled from meters

        Returns
            dict: Dictionary of metric name and value, one for each
            `meters` that made at least one measure
        """
        return {**self.prepend_name_dict(self._prefixes[0], self._train_metrics),
                **self.prepend_name_dict(self._prefixes[1], self._val_metrics)}

    @property
    def train_metrics(self):
        return dict(self._train_metrics)

    @property
    def val_metrics(self):
        return dict(self._val_metrics)

    def _compile_train_metrics(self):
        self._train_metrics = {}

        for metric_name, meter in self.train_meters.items():
            try:
                value = meter.value()
                self._train_metrics[metric_name] = value
            except ZeroMeasurementsError:
                continue

    @property
    def epochs_trained(self):
        """ Total number of epochs epochs_trained

        Returns:
            int: number of epochs
        """
        return self._epochs_trained

    @property
    def steps_trained(self):
        return self._steps_trained

    @epochs_trained.setter
    def epochs_trained(self, value):
        if value < 0:
            raise AttributeError('can\'t set epochs_trained'
                                 'to a value less than zero')

    @abstractmethod
    def update_batch(self, *args, **kwargs):
        """ Abstract method for update model parameters given a batch

        Args:
            *args (variable length arguments of
            :class:`torch.autograd.Variable` of Tensors or cuda Tensors):
                Unamed batch parameters
            **kwargs (variable length keyword arguments of
                      :class:`torch.autograd.Variable` of
                      Tensors or cuda Tensors):
                Named batch parameters
        """
        pass

    def reset_meters(self):
        self._train_metrics = {}
        self._val_metrics = {}
        for meter in self.train_meters.values():
            meter.reset()

    def _prepare_tensor(self, x):
        if torch.is_tensor(x):
            return Variable(self._convert_tensor(x))
        else:
            return x

    def log(self):
        self._callbacks.on_log()

    def log_started(self):
        return (self.logging_frecuency > 0 and
                self.step % self.logging_frecuency == 0)

    def _train_epoch(self, train_dataloader, valid_dataloader=None):
        self.reset_meters()
        for self.step, batch in enumerate(train_dataloader):
            # convert to 1-d tuple if batch was a tensor instead of a tuple
            if torch.is_tensor(batch):
                batch = (batch, )
            batch = map(self._prepare_tensor, batch)
            self.update_batch(*batch)

            self._steps_trained += 1

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

        self._raised_stop_training = False
        self.total_epochs = epochs
        self.total_steps = len(dataloader)
        self.valid_dataloader = valid_dataloader

        self._callbacks.on_train_begin()

        # Turn model to training mode
        self.model.train(mode=True)

        self.epoch = 0
        while (self.epoch < self.total_epochs
                and not self._raised_stop_training):
            self._callbacks.on_epoch_begin()
            self._train_epoch(dataloader, valid_dataloader)
            self._callbacks.on_epoch_end()
            self.epoch += 1

        self._callbacks.on_train_end()

        del self.valid_dataloader

        # Turn model to evaluation mode
        self.model.train(mode=False)

    def _is_time_to_log(self):
        log_frec = self.logging_frecuency
        return (log_frec > 0 and
                ((self.total_steps % log_frec != 0 and
                 self.step == self.total_steps - 1)
                 or self.step % log_frec == log_frec - 1))

    def _validate(self):
        self._val_metrics = self.validator.validate(self.valid_dataloader)

    @property
    def val_meters(self):
        """ Returns the list of Meters for the validation set
        """
        return self.validator.meters

    @property
    def hparams(self):
        """ Returns a dictionary of <hyper-parameter name, hyper-parameter value
        """
        return self._hparams

    def evaluate(self, dataloader, metrics=None):
        if metrics is not None:
            metrics = parse_meters(metrics)
        else:
            metrics = self.val_meters
        validator = self.create_validator(metrics)
        results = validator.validate(dataloader)
        self.model.train(mode=False)
        return results

    def stop_training(self):
        self._raised_stop_training = True

    def add_named_train_meter(self, name, meter):
        if name in self.train_meters:
            raise Exception(self.METER_ALREADY_EXISTS_MESSAGE
                                .format(name=name))

        self.train_meters[name] = meter

    def add_named_val_meter(self, name, meter):
        self.validator.add_named_meter(name, meter)

    def add_callback(self, callback):
        self._callbacks.add(callback)

    def _create_hparams(self, hparams):
        result = {}
        for param_name, obj in hparams.items():
            if isinstance(obj, torchero.hparams.P):
                result[param_name] = obj
            elif hasattr(obj, '__call__'):
                result[param_name] = torchero.hparams.LambdaP(obj)
            else:
                result[param_name] = torchero.hparams.FixedP(obj)
            result[param_name].accept(self)
        result = ParamsDict(result)
        return result
