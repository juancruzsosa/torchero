from .base import BaseTrainer

class SupervisedTrainer(BaseTrainer):
    """ SupervisedTrainer
    """

    def __init__(self, model, criterion, optimizer, logging_frecuency=1):
        """ Constructor

        Args:
            model (instance of :model:`torch.nn.Module`): Model to train
            criterion (:model:`torch.nn.Module`): Loss criterion (eg: `torch.nn.CrossEntropyLoss`, `torch.nn.L1Loss`)
            optimizer (instance of :model:`torch.optim.Optimizer`): Model Optimizer
            logging_frecuency (int): Frecuency of log to monitor train/validation
        """
        super(SupervisedTrainer, self).__init__(model=model, logging_frecuency=logging_frecuency)
        self.criterion = criterion
        self.optimizer = optimizer

    def update_batch(self, x, y):
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
