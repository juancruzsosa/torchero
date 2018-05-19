from .base import BatchTrainer, BatchValidator, ValidationGranularity
from .meters import Averager, NullMeter

class SupervisedValidator(BatchValidator):
    def __init__(self, model, meters, criterion):
        super(SupervisedValidator, self).__init__(model, meters)
        self.criterion = criterion

    def validate_batch(self, x, y):
        output = self.model(x)
        loss = self.criterion(output, y)

        self._meters['val_loss'].measure(loss.data[0])
        self._meters['val_acc'].measure(output.data, y.data)

class SupervisedTrainer(BatchTrainer):
    """ Supervised trainer
    """

    def create_validator(self):
        return SupervisedValidator(self.model, self.val_meters, self.criterion)

    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 callbacks=[],
                 acc_meter=NullMeter(),
                 val_acc_meter=None,
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
            acc_meter (:class: `torchtrainer.meters.Meter'):
                Training accuracy meter
            val_acc_meter (:class: `torchtrainer.meters.Meter'):
                Validation accuracy meter
            logging_frecuency (int):
                Frecuency of log to monitor train/validation
            validation_granularity (ValidationGranularity):
                Change validation criterion (after every log vs after every epoch)
        """
        if val_acc_meter is None:
            val_acc_meter = acc_meter.clone()

        train_meters = {'train_loss' : Averager(),
                        'train_acc' : acc_meter}

        val_meters = {'val_loss': Averager(),
                      'val_acc': val_acc_meter}

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

        self.train_meters['train_loss'].measure(loss.data[0])
        self.train_meters['train_acc'].measure(output.data, y.data)
