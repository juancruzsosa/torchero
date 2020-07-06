import torch
from .base import BatchTrainer, BatchValidator, ValidationGranularity
from .meters import LossMeter
from .utils.defaults import get_optimizer_by_name, get_loss_by_name

class SupervisedValidator(BatchValidator):
    def __init__(self, model, meters):
        super(SupervisedValidator, self).__init__(model, meters)

    def validate_batch(self, x, y):
        output = self.model(x)

        for meter in self.meters.values():
            meter.measure(output.data, y.data)


class SupervisedTrainer(BatchTrainer):
    """ Supervised trainer
    """

    def create_validator(self):
        return SupervisedValidator(self.model, self.val_meters)

    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 callbacks=[],
                 acc_meters={},
                 val_acc_meters=None,
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
            logging_frecuency (int):
                Frecuency of log to monitor train/validation
            prefixes (tuple, list):
                Prefixes of train and val metrics
            validation_granularity (ValidationGranularity):
                Change validation criterion (after every log vs after every epoch)
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
                                                callbacks=callbacks,
                                                logging_frecuency=logging_frecuency,
                                                prefixes=prefixes,
                                                validation_granularity=validation_granularity)

        self.add_named_train_meter('loss', LossMeter(criterion))
        self.add_named_val_meter('loss', LossMeter(criterion))

    def update_batch(self, x, y):
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

        self.model.train(mode=False)
        with torch.no_grad():
            for meter in self.train_meters.values():
                meter.measure(output.data, y.data)
        self.model.train(mode=True)
