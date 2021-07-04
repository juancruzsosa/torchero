class Callback(object):
    """ Callbacks for epoch/batch training events
    """
    def __init__(self):
        """ Constructor
        """
        self.trainer = None

    def accept(self, trainer):
        """ Attach the callback to the given trainer

        Args:
            trainer(instance of :class:`torchero.base.BaseTrainer`):
                Trainer to attach to
        """
        self.trainer = trainer

    def on_epoch_begin(self):
        """ Method invoked before every epoch
        """
        pass

    def on_epoch_end(self):
        """ Method invoked after every epoch
        """
        pass

    def on_log(self):
        """ Method invoked after every log update
        """
        pass

    def on_train_begin(self):
        """ Method invoked at the beginning of the training
        """
        pass

    def on_train_end(self):
        """ Method invoked at the end of the training
        """
        pass
