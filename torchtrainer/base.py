import torch
from torch.autograd import Variable
from abc import abstractmethod
from .hooks import HookContainer
from .meters import ZeroMeasurementsError


class BaseTrainer(object):
    """ Base Trainer for all Trainer classes.
        All trainers should subclass this class
    """

    INVALID_EPOCH_MESSAGE='Expected epoch to be a non-negative integer, got: {epochs}'
    INVALID_LOGGING_FRECUENCY_MESSAGE='Expected loggin frecuency to be a non-negative integer, got: {logging_frecuency}'

    def __init__(self, model, hooks=[], logging_frecuency=1):
        """ Constructor

        Args:
            model (:class:`torch.nn.Module`): Module to train
            hooks (:class:`torchtrainer.hooks.Hook`): Pluggable hooks for epoch/batch events.
            logging_frecuency (int): Frecuency of log to monitor train/validation
        """
        if logging_frecuency < 0:
            raise Exception(self.INVALID_LOGGING_FRECUENCY_MESSAGE.format(logging_frecuency=logging_frecuency))
        self.logging_frecuency = logging_frecuency

        self.model = model
        self._epochs_trained = 0
        self._use_cuda = False
        self._last_stats = {}
        self.stats_meters = {}

        self._hooks = HookContainer(self)
        for hook in hooks:
            self._hooks.attach(hook)

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

    @property
    def last_stats(self):
        """ Last statistic recopiled from stats_meters

        Returns
            dict: Dictionary of metric name and value, one for each
            `stats_meters` that made at least one measure
        """
        return self._last_stats

    def _compile_last_stats(self):
        self._last_stats = {}
        for metric_name, meter in self.stats_meters.items():
            try:
                self._last_stats[metric_name]  = meter.value()
            except ZeroMeasurementsError:
                continue

    @property
    def epochs_trained(self):
        """ Total number of epochs epochs_trained

        Returns:
            int: number of epochs
        """
        return self._epochs_trained

    @abstractmethod
    def update_batch(self, *args, **kwargs):
        """ Abstract method for update model parameters given a batch

        Args:
            *args (variable length arguments of :class:`torch.autograd.Variable` of Tensors or cuda Tensors): Unamed batch parameters
            **kwargs (variable length keyword arguments of :class:`torch.autograd.Variable` of Tensors or cuda Tensors): Named batch parameters
        """

        pass

    def log(self):
        self._compile_last_stats()
        self._hooks.log()
        for meter in self.stats_meters.values():
            meter.reset()

    def _train_epoch(self, train_dataloader, valid_dataloader=None):
        for self.step, batch in enumerate(train_dataloader):
            # convert to 1-d tuple if batch was a tensor instead of a tuple
            if torch.is_tensor(batch):
                batch = (batch, )
            batch = list(map(self._to_variable, batch))
            self.update_batch(*batch)

            if self._is_time_to_log():
                self.model.train(mode=False)
                if valid_dataloader:
                    self._validate(valid_dataloader)
                self.log()
                self.model.train(mode=True)

        self._epochs_trained += 1

    def train(self, dataloader, valid_dataloader=None, epochs=1):
        """ Train the model

        Args:
            dataloader (:class:`torch.utils.DataLoader`): Train data loader
            valid_dataloader (:class:`torch.utils.DataLoader`): Validation data loader
            epochs (int): Number of epochs to train
        """
        if epochs < 0:
            raise Exception(self.INVALID_EPOCH_MESSAGE.format(epochs=epochs))

        self.total_epochs = epochs
        self.total_steps = len(dataloader)

        # Turn model to training mode
        self.model.train(mode=True)

        for self.epoch in range(self.total_epochs):
            self._hooks.pre_epoch()
            self._train_epoch(dataloader, valid_dataloader)
            self._hooks.post_epoch()

        # Turn model to evaluation mode
        self.model.train(mode=False)

    def _is_time_to_log(self):
        return (self.logging_frecuency > 0 and
                self.step % self.logging_frecuency == self.logging_frecuency - 1)

    @abstractmethod
    def validate_batch(self, *arg, **kwargs):
        """ Abstract method for validate model per batch

        Args:
            *args (variable length arguments of :class:`torch.autograd.Variable` of Tensors or cuda Tensors): Unamed batch parameters
            **kwargs (variable length keyword arguments of :class:`torch.autograd.Variable` of Tensors or cuda Tensors): Named batch parameters
        """
        pass

    def _validate(self, valid_dataloader):
        for valid_batch in valid_dataloader:
            if isinstance(valid_batch, torch.Tensor):
                valid_batch = (valid_batch, )
            valid_batch = list(map(self._to_variable, valid_batch))
            self.validate_batch(*valid_batch)