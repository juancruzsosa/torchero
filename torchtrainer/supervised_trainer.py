from .base import BatchTrainer, BatchValidator, ValidationGranularity
from .meters import LossMeter

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

    @staticmethod
    def prepend_name_dict(prefix, d):
        return {prefix + name: value for name, value in d.items()}

    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 callbacks=[],
                 acc_meters={},
                 val_acc_meters=None,
                 logging_frecuency=1,
                 validation_granularity=ValidationGranularity.AT_EPOCH):
        """ Constructor

        Args:
            model (instance of :model:`torch.nn.Module`):
                Model to train
            criterion (:model:`torch.nn.Module`):
                Loss criterion (eg: `torch.nn.CrossEntropyLoss`,
                                    `torch.nn.L1Loss`)
            callbacks (:class:`torchtrainer.callbacks.Callback`):
                Pluggable callbacks for epoch/batch events.
            optimizer (instance of :model:`torch.optim.Optimizer`):
                Model optimizer
            acc_meters (dictionary of :class:`torchtrainer.meters.Meter'):
                Training accuracy meters by meter name
            val_acc_meters (dictionary of :class:`torchtrainer.meters.Meter'):
                Validation accuracy meter by meter name
            logging_frecuency (int):
                Frecuency of log to monitor train/validation
            validation_granularity (ValidationGranularity):
                Change validation criterion (after every log vs after every epoch)
        """
        if val_acc_meters is None:
            val_acc_meters = {name: meter.clone() for name, meter in acc_meters.items()}

        train_meters = {'loss' : LossMeter(criterion),
                        **acc_meters}

        val_meters = {'loss': LossMeter(criterion),
                      **val_acc_meters}

        train_meters = self.prepend_name_dict('train_', train_meters)
        val_meters = self.prepend_name_dict('val_', val_meters)

        self.criterion = criterion
        self.optimizer = optimizer

        super(SupervisedTrainer, self).__init__(model=model,
                                                train_meters=train_meters,
                                                val_meters=val_meters,
                                                callbacks=callbacks,
                                                logging_frecuency=logging_frecuency,
                                                validation_granularity=validation_granularity)

    def update_batch(self, x, y):
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

        for meter in self.train_meters.values():
            meter.measure(output.data, y.data)
