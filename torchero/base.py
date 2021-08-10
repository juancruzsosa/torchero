import pickle
import json
import logging

from abc import ABCMeta, abstractmethod
from enum import Enum

import torch

import torchero.hparams
from torchero.callbacks import Callback, CallbackContainer, History, ModelCheckpoint
from torchero.utils.defaults import parse_meters
from torchero.utils.mixins import DeviceMixin
from torchero.utils.collections import MetricsDict, ParamsDict


class ValidationGranularity(Enum):
    AT_LOG = 'log' #validate after log event (more granular display but less performant)
    AT_EPOCH = 'epoch' #validate after each epoch (less granular display but more performant)


class _OnLogValidScheduler(Callback):
    """ Callback to validate responsible after each log event
    This callback is not intended to be used explicitly
    """
    def on_log(self):
        self.trainer._validate()


class _OnEpochValidScheduler(Callback):
    """ Callback responsible to validate after each epoch
    """
    def on_log(self):
        if self.trainer.step == self.trainer.total_steps-1:
            self.trainer._validate()

class BatchValidator(DeviceMixin, metaclass=ABCMeta):
    """ Abstract class for all validation classes that works with batched inputs.
        All those validators should subclass this class.
        This class handles the computation of the metrics
        for the validation data
    """

    def __init__(self, model, meters):
        """ Constructor

        Argument:
            model (``torch.nn.Module``): Module to be trained
            meters (list/dict of meters): List/Dict of metrics to evaluate on
                the model. If a list is passed the metrics names will be auto
                completed. If
        """
        super(BatchValidator, self).__init__()
        self.model = model
        self._metrics = MetricsDict(parse_meters(meters))

    @abstractmethod
    def validate_batch(self, *arg, **kwargs):
        """ Validate model

        Args:
            *args (variable length arguments
                   of :class:`torch.autograd.Variable`
                   of Tensors or cuda Tensors):
                Unnamed batch parameters
            **kwargs (variable length keyword arguments of
                      :class:`torch.autograd.Variable` of
                      Tensors or cuda Tensors):
                Named batch parameters
        """
        pass

    def meters_names(self):
        """ Returns the metrics names
        """
        return self.meters.keys()

    @property
    def meters(self):
        """ Returns the meters associated to the given metrics
        """
        return self._metrics.meters

    @property
    def metrics(self):
        """ Returns the list of metrics of the last validation
        """
        return self._metrics

    def validate(self, valid_dataloader):
        """ Validate a model against a given validation dataloader

        Arguments:
            valid_dataloader: Input dataloader

        Returns:
            A dictionary of metrics for the given dataloader
        """
        self._metrics.reset()

        if not valid_dataloader:
            return self._metrics

        self.model.train(mode=False)
        with torch.no_grad():
            for batch in valid_dataloader:
                if isinstance(batch, torch.Tensor):
                    batch = (batch, )
                batch = map(self._prepare_tensor, batch)
                self.validate_batch(*batch)
        self.model.train(mode=True)

        self._metrics.compile()
        return dict(self._metrics)

    def validate_all(self, dataloaders):
        """ Validate the model against multiple dataloaders

        Arguments:
            validate_all (dict): Dataloaders (dict values) by name (dict keys)

        Returns:
            A dictionary of all the metrics (dict) for each dataloader name
        """
        records = {split_name: self.validate(dl) for split_name, dl in dataloaders.items()}
        return records

    def add_named_meter(self, name, meter):
        """ Add a meter with name
        """
        self._metrics.add_metric(name, meter)

    def _save_to_zip(self, zip_fp, prefix):
        """ Save the validator state from a zip handler
        """
        with zip_fp.open(prefix + '/val_metrics.pkl', 'w') as metrics_fp:
            pickle.dump(self._metrics, metrics_fp)

    def _load_from_zip(self, zip_fp, prefix=''):
        """ Load the validator state from a zip handler
        """
        with zip_fp.open(prefix + '/val_metrics.pkl', 'r') as metrics_fp:
            self._metrics = pickle.load(metrics_fp)


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
    logger = None

    @classmethod
    def create_default_logger_handler(cls):
        """ Set up a logger to be shared across all instances
        """
        logger = logging.getLogger("Trainer")
        logger.setLevel(logging.INFO)
        cls.logger = logger


    @staticmethod
    def prepend_name_dict(prefix, d):
        """ Add a prefix to each key of the given dictionary

        Arguments:
            prefix (str): Prefix to add to every key
            d (dict): Input dictionary

        Returns:
            A dictionary with the keys prefixed
        """
        return {prefix + name: value for name, value in d.items()}

    @abstractmethod
    def create_validator(self, metrics):
        """ Returns the most appropriate ValidatorClass
        """
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
            callbacks (list of :class:`torchero.callbacks.Callback`):
                List of callbacks to use during training.
            train_meters (list or dict of :class: `torchero.meters.Meter'):
                Training meters
            val_meters (list or dict of :class: `torchero.meters.Meter'):
                Validation meters
            hparams (dict of hyperparameters):
                Dictionary <name,hparam> of hyperparameters. Each value
                can be a fixed value, a lambda function with only parameter to pass to the trainer,
                or a instance of `torchero.hparams.OptimP`
            logging_frecuency (int):
                Frecuency of log events in training steps units
            prefixes (tuple, list):
                Prefixes of train and val metrics to avoid metrics names collisions
            validation_granularity (ValidationGranularity):
                Set to ValidationGranularity.AT_LOG to validate after log event
                    (more granular display but less performant),
                Set to ValidationGranularity.AT_EPOCH to validate after each
                    epoch (less granular display but more performant)
        """
        if logging_frecuency < 0:
            raise ValueError(self.INVALID_LOGGING_FRECUENCY_MESSAGE
                                 .format(logging_frecuency=logging_frecuency))

        if not isinstance(validation_granularity, ValidationGranularity) \
                or validation_granularity not in ValidationGranularity:
            raise ValueError(self.INVALID_VALIDATION_GRANULARITY_MESSAGE
                                 .format(mode=validation_granularity))

        super(BatchTrainer, self).__init__()

        valid_sched = self.SCHED_BY_GRANULARITY[validation_granularity]()

        self.logging_frecuency = logging_frecuency

        self.create_default_logger_handler()

        self.model = model

        # Initialize training stats
        self._epochs_trained = 0
        self._steps_trained = 0
        self._prefixes = prefixes

        # Set up metrics
        train_meters = parse_meters(train_meters)

        if val_meters is None:
            val_meters = {name: meter.clone()
                          for name, meter in train_meters.items()}
        else:
            val_meters = parse_meters(val_meters)

        self._train_metrics = MetricsDict(train_meters)

        # Set up hyper-parameters
        self._hparams = self._create_hparams(dict(hparams))

        # Set up validator
        self.validator = self.create_validator(val_meters)

        # Flag to determine if the training has to be
        # stopped (e.g. due to EarlyStopping)
        self._raised_stop_training = False

        self._history_callback = History()

        self._callbacks = CallbackContainer()
        self._callbacks.accept(self)

        # Predefined callbacks
        # Validation callback to validate after the defined event
        self._callbacks.add(valid_sched)
        # History callback to record every stat
        self._callbacks.add(self._history_callback)

        # Add user defined callbacks
        for callback in callbacks:
            self._callbacks.add(callback)

    def _save_to_zip(self, zip_fp, prefix=''):
        """ Dumps the full training state to a zip handler

        Arguments:
            zip_fp (``zipfile.ZipFile``): ZipFile object to dump the state
            prefix (str): Trainer folder in the zip
        """
        prefix = prefix.rstrip('/')
        # Dump training state
        with zip_fp.open(prefix + '/config.json', 'w') as config_fp:
            config_fp.write(json.dumps(self.config, indent=4).encode())
        # Dump validator (which saves the validation metrics)
        self.validator._save_to_zip(zip_fp, prefix=prefix)
        # Dump training metrics results and trainers
        with zip_fp.open(prefix + '/train_metrics.pkl', 'w') as metrics_fp:
            pickle.dump(self._train_metrics, metrics_fp)
        # Dump list of callbacks
        with zip_fp.open(prefix + '/callbacks.pkl', 'w') as callbacks_fp:
            pickle.dump(self._callbacks, callbacks_fp)

    def _load_from_zip(self, zip_fp, prefix=''):
        """ Loads the full training state from a zip handler

        Arguments:
            zip_fp (``zipfile.ZipFile``): ZipFile object to load the state
            prefix (str): Trainer folder in the zip
        """
        prefix = prefix.rstrip('/')
        # Load training state from config
        with zip_fp.open(prefix + '/config.json', 'r') as config_fp:
            config = json.loads(config_fp.read().decode())
            self._epochs_trained = config['epochs_trained']
            self._steps_trained = config['steps_trained']
            self._logging_frequency = config['logging_frequency']
            self._prefixes = config['prefixes']
            self._raised_stop_training = config['raised_stop_training']
        # Load validator
        self.validator._load_from_zip(zip_fp, prefix=prefix)
        with zip_fp.open(prefix + '/train_metrics.pkl', 'r') as metrics_fp:
            self._train_metrics = pickle.load(metrics_fp)
        with zip_fp.open(prefix + '/callbacks.pkl', 'r') as callbacks_fp:
            callbacks = pickle.load(callbacks_fp)
            self._callbacks = CallbackContainer()
            self._callbacks.accept(self)
            for callback in callbacks:
                self._callbacks.add(callback)

    @property
    def config(self):
        """ Training state counters
        """
        return {'epochs_trained': self._epochs_trained,
                'steps_trained': self._steps_trained,
                'logging_frequency': self.logging_frecuency,
                'prefixes': self._prefixes,
                'raised_stop_training': self._raised_stop_training}

    @property
    def history(self):
        """ Historical training metrics
        """
        return self._history_callback.registry

    def to(self, device):
        """ Moves the trainer to a device

        Arguments:
            device (str or ``torch.device``)
        """
        super(BatchTrainer, self).to(device)
        self.model.to(self._device)
        self.validator.to(self._device)

    def meters_names(self):
        """ Returns the meters names
        """
        return sorted(self.meters.keys())

    @property
    def meters(self):
        """ Returns both training and validation meters
        """
        return {**self.prepend_name_dict(self._prefixes[0], self._train_metrics.meters),
                **self.prepend_name_dict(self._prefixes[1], self.validator.meters)}

    @property
    def metrics(self):
        """ Last calculated metrics for training & validation data
        """
        return {**self.prepend_name_dict(self._prefixes[0], self._train_metrics),
                **self.prepend_name_dict(self._prefixes[1], self.validator.metrics)}

    @property
    def train_metrics(self):
        """ Metrics results of only the training data
        """
        return dict(self._train_metrics)

    @property
    def val_metrics(self):
        """ Metrics results of only the training data
        """
        return dict(self.validator.metrics)

    @property
    def epochs_trained(self):
        """ Total number of epochs trained
        """
        return self._epochs_trained

    @property
    def steps_trained(self):
        """ Total number of trained iterations
        """
        return self._steps_trained

    @epochs_trained.setter
    def epochs_trained(self, value):
        if value < 0:
            raise AttributeError('can\'t set epochs_trained'
                                 'to a value less than zero')

    @abstractmethod
    def update_batch(self, *args, **kwargs):
        """ Update model parameters based on a given loss with the current batch

        Args:
            *args (variable length arguments of
            :class:`torch.autograd.Variable` of Tensors or cuda Tensors):
                Unnamed batch parameters
            **kwargs (variable length keyword arguments of
                      :class:`torch.autograd.Variable` of
                      Tensors or cuda Tensors):
                Named batch parameters
        """
        pass

    def log(self):
        """ Broadcast a logging event to all the callbacks
        """
        self._callbacks.on_log()

    def log_started(self):
        """ Returns if it's time to log
        """
        return (self.logging_frecuency > 0 and
                self.step % self.logging_frecuency == 0)

    def _train_epoch(self, train_dataloader, valid_dataloader=None):
        """ Train the model & validate the model for one epoch

        Arguments:
            train_dataloader (:class:`torch.utils.DataLoader`):
                Train data loader
            valid_dataloader (:class:`torch.utils.DataLoader`):
                Validation data loader
        """
        self._train_metrics.reset()
        self.validator.metrics.reset()

        for self.step, batch in enumerate(train_dataloader):
            # convert to 1-d tuple if batch was a tensor instead of a tuple
            if torch.is_tensor(batch):
                batch = (batch, )
            batch = map(self._prepare_tensor, batch)
            self.update_batch(*batch)

            self._steps_trained += 1

            if self._is_time_to_log():
                self._train_metrics.compile()
                self.log()

        self._epochs_trained += 1

    def train(self, dataloader, valid_dataloader=None, epochs=1):
        """ Train and validates the model

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
        """ Returns if it's time to log
        """
        log_frec = self.logging_frecuency
        return (log_frec > 0 and
                ((self.total_steps % log_frec != 0 and
                 self.step == self.total_steps - 1)
                 or self.step % log_frec == log_frec - 1))

    def _validate(self):
        """ Validate the model against the validation dataloader
        """
        self.validator.validate(self.valid_dataloader)

    @property
    def train_meters(self):
        """ Meters for training data
        """
        return self._train_metrics.meters

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
        validator.to(self._device)
        results = validator.validate(dataloader)
        self.model.train(mode=False)
        return results

    def evaluate_all(self, dataloaders, metrics=None):
        if metrics is not None:
            metrics = parse_meters(metrics)
        else:
            metrics = self.val_meters
        validator = self.create_validator(metrics)
        validator.to(self._device)
        results = validator.validate_all(dataloaders)
        self.model.train(mode=False)
        return results

    def stop_training(self):
        self._raised_stop_training = True

    def add_named_train_meter(self, name, meter):
        self._train_metrics.add_metric(name, meter)

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

    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoints = []
            for callback in self._callbacks:
                if isinstance(callback, ModelCheckpoint):
                    checkpoints.append(callback)
            if len(checkpoints) == 0:
                raise Exception("No ModelCheckpoint callback found")
            elif len(checkpoints) > 1:
                raise Exception("Multiple checkpoints found. Pass the checkpoint to load")
            else: # len(checkpoints) == 1
                checkpoint = checkpoints[0]

        checkpoint.load()

    @property
    def callbacks(self):
        return list(self._callbacks)
