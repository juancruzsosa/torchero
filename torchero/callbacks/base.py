class Callback(object):
    """ Callbacks for epoch/batch training events
    """
    def __init__(self):
        """ Constructor
        """
        self.trainer = None

    def accept(self, trainer):
        """ Accept a trainer

        Args:
            trainer(instance of :class:`torchero.base.BaseTrainer`):
                Trainer to attach to
        """
        self.trainer = trainer

    def on_epoch_begin(self):
        """ Trigger called before every epoch
        """
        pass

    def on_epoch_end(self):
        """ Trigger called after every epoch
        """
        pass

    def on_log(self):
        """ Trigger called after every log update
        """
        pass

    def on_train_begin(self):
        """ Trigger called one time before training
        """
        pass

    def on_train_end(self):
        """ Trigger called one time after training
        """
        pass
