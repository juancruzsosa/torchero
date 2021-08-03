import torch
from torch import nn

from torchero.base import BatchTrainer, BatchValidator, ValidationGranularity
from torchero.meters import LossMeter
from torchero.utils.defaults import get_loss_by_name, get_optimizer_by_name
from torchero.utils.optimizer import optimizer_to


class SupervisedValidator(BatchValidator):
    """ Class for evaluating torch models on validation
    datasets
    """
    def __init__(self, model, meters):
        super(SupervisedValidator, self).__init__(model, meters)

    def validate_batch(self, x, y):
        if isinstance(x, (tuple, list)):
            output = self.model(*x)
        else:
            output = self.model(x)

        self._metrics.measure(output.data, y.data)


class SupervisedTrainer(BatchTrainer):
    """ Class for training torch models on labeled data
    """
    def create_validator(self, metrics):
        return SupervisedValidator(self.model, metrics)

    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 callbacks=[],
                 acc_meters={},
                 val_acc_meters=None,
                 hparams={},
                 logging_frecuency=1,
                 prefixes=('train_', 'val_'),
                 validation_granularity=ValidationGranularity.AT_EPOCH):
        """ Constructor

        Args:
            model (instance of :model:`torch.nn.Module`):
                Model to train
            criterion (:model:`torch.nn.Module`):
                Loss criterion (eg: `torch.nn.CrossEntropyLoss`,
                                    `torch.nn.L1Loss`)
            callbacks (:class:`torchero.callbacks.Callback`):
                Pluggable callbacks for epoch/batch events.
            optimizer (instance of :model:`torch.optim.Optimizer`):
                Model optimizer
            acc_meters (dictionary of :class:`torchero.meters.Meter'):
                Training accuracy meters by meter name
            val_acc_meters (dictionary of :class:`torchero.meters.Meter'):
                Validation accuracy meter by meter name
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
        if isinstance(criterion, str):
            criterion = get_loss_by_name(criterion)
        self.criterion = criterion

        if isinstance(optimizer, str):
            optimizer = get_optimizer_by_name(optimizer, model)
        self.optimizer = optimizer

        super(SupervisedTrainer, self).__init__(model=model,
                                                train_meters=acc_meters,
                                                val_meters=val_acc_meters,
                                                hparams=hparams,
                                                callbacks=callbacks,
                                                logging_frecuency=logging_frecuency,
                                                prefixes=prefixes,
                                                validation_granularity=validation_granularity)

        self.add_named_train_meter('loss', LossMeter(criterion))
        self.add_named_val_meter('loss', LossMeter(criterion))

    def update_batch(self, x, y):
        self.optimizer.zero_grad()
        if isinstance(x, (tuple, list)):
            output = self.model(*x)
        else:
            output = self.model(x)
        if isinstance(self.criterion, (nn.BCEWithLogitsLoss, nn.BCELoss)):
            y = y.double()
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

        self.model.train(mode=False)
        self._train_metrics.measure(output.data, y.data)
        self.model.train(mode=True)

    def __repr__(self):
        return "{cls}(model={model},\n\
    criterion={criterion},\n\
    optimizer={optimizer},\n\
    callbacks={callbacks},\n\
    acc_meters={train_meters},\n\
    val_acc_meters={val_meters},\n\
    hparams={hparams},\n\
    logging_frecuency={logging_frecuency},\n\
    prefixes={prefixes})".format(
            cls=self.__class__.__name__,
            model=repr(self.model),
            criterion=repr(self.criterion),
            optimizer=repr(self.optimizer),
            callbacks=repr(self._callbacks),
            train_meters=repr(self._train_metrics.meters),
            val_meters=repr(self.validator.meters),
            hparams=repr(self._hparams),
            logging_frecuency=repr(self.logging_frecuency),
            prefixes=repr(self._prefixes))

    def to(self, device):
        super(SupervisedTrainer, self).to(device)
        self.criterion.to(self._device)
        optimizer_to(self.optimizer, self._device)

    def _save_to_zip(self, zip_fp, prefix=''):
        prefix = prefix.rstrip('/')
        super(SupervisedTrainer, self)._save_to_zip(zip_fp, prefix=prefix)
        with zip_fp.open(prefix + '/loss.pkl', 'w') as loss_fp:
            torch.save(self.criterion, loss_fp)
        with zip_fp.open(prefix + '/optimizer.pkl', 'w') as optim_fp:
            torch.save(self.optimizer, optim_fp)

    def _load_from_zip(self, zip_fp, prefix=''):
        prefix = prefix.rstrip('/')
        super(SupervisedTrainer, self)._load_from_zip(zip_fp, prefix=prefix)
        with zip_fp.open(prefix + '/loss.pkl', 'r') as loss_fp:
            self.criterion = torch.load(loss_fp)
        with zip_fp.open(prefix + '/optimizer.pkl', 'r') as optim_fp:
            self.optimizer = torch.load(optim_fp)
