from .base import BatchTrainer
from .meters import Averager, NullMeter

class SupervisedTrainer(BatchTrainer):
    """ Supervised trainer
    """

    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 acc_meter=NullMeter(),
                 val_acc_meter=None,
                 callbacks=[],
                 logging_frecuency=1):
        """ Constructor

        Args:
            model (instance of :model:`torch.nn.Module`):
                Model to train
            criterion (:model:`torch.nn.Module`):
                Loss criterion (eg: `torch.nn.CrossEntropyLoss`,
                                    `torch.nn.L1Loss`)
            optimizer (instance of :model:`torch.optim.Optimizer`):
                Model optimizer
            logging_frecuency (int):
                Frecuency of log to monitor train/validation
        """
        super(SupervisedTrainer, self).__init__(model=model,
                                                callbacks=callbacks,
                                                logging_frecuency=logging_frecuency)
        self.criterion = criterion
        self.optimizer = optimizer

        if val_acc_meter is None:
            val_acc_meter = acc_meter.clone()

        self.meters['train_loss'] = Averager()
        self.meters['val_loss'] = Averager()
        self.meters['train_acc'] = acc_meter
        self.meters['val_acc'] = val_acc_meter

    def update_batch(self, x, y):
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

        self.meters['train_loss'].measure(loss.data[0])
        self.meters['train_acc'].measure(output.data, y.data)

    def validate_batch(self, x, y):
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)

        self.meters['val_loss'].measure(loss.data[0])
        self.meters['val_acc'].measure(output.data, y.data)
