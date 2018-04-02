import torch
from torch.autograd import Variable
from abc import abstractmethod

class BaseTrainer(object):
    """ Base Trainer for all Trainer classes.
        All trainers should subclass this class
    """

    INVALID_EPOCH_MESSAGE='Expected epoch to be a non-negative integer, got: {epochs}'

    def __init__(self, model):
        """ Constructor

        Args:
            model (:class:`torch.nn.Module`): Module to train
        """
        self.model = model
        self._epochs_trained = 0

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

    def _train_epoch(self, train_dataloader):
        for self.step, batch in enumerate(train_dataloader):
            # convert to 1-d tuple if batch was a tensor instead of a tuple
            if torch.is_tensor(batch):
                batch = (batch, )
            batch = list(map(Variable, batch))
            self.update_batch(*batch)

        self._epochs_trained += 1

    def train(self, dataloader, epochs=1):
        """ Train the model

        Args:
            dataloader (:class:`torch.utils.DataLoader`): Train data loader
            epochs (int): Number of epochs to train
        """
        if epochs < 0:
            raise Exception(self.INVALID_EPOCH_MESSAGE.format(epochs=epochs))

        self.total_epochs = epochs
        self.total_steps = len(dataloader)

        # Turn model to training mode
        self.model.train(mode=True)

        for self.epoch in range(self.total_epochs):
            self._train_epoch(dataloader)

        # Turn model to evaluation mode
        self.model.train(mode=False)
