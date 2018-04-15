class Hook(object):
    """ Hooks for epoch/batch training events
    """
    def __init__(self):
        """ Constructor
        """
        self.trainer = None

    def accept(self, trainer):
        """ Accept a trainer

        Args:
            trainer(instance of :class:`torchtrainer.base.BaseTrainer`): Trainer to attach to
        """
        self.trainer = trainer

    def pre_epoch(self):
        """ Trigger called before every epoch
        """
        pass

    def post_epoch(self):
        """ Trigger called after every epoch
        """
        pass

    def log(self):
        """ Trigger called after every log update
        """
        pass
